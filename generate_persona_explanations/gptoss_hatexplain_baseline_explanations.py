import json
import os
import re
import time
import logging
import argparse
from openai import OpenAI

from rationale_binary_utils import compute_rationale_binary

# ========== CONFIGURATION ==========
API_KEY = "<YOUR_API_KEY>" 
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-oss-120b"
TEST_MODE = False  # Set to False for full run

# ==== LOGGING ====
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/generate_baseline_explanations_gptoss.log", mode='a', encoding='utf-8')
    ]
)

# ========== QWEN CLIENT ==========
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# ========== DATA LOCATIONS ==========
QUESTIONS_PATH = "../datasets/personas_&_questions/hatexplain_500_hybrid.jsonl"
RESULTS_DIR = "../datasets/results_gpt_oss_120b_500_r1_medium"
OUTPUT_NAME = "baseline_gpt_oss_120b.jsonl"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCALE = ["Hate speech", "Offensive language", "Normal"]

# ========== HELPER FUNCTIONS ==========

def load_existing_results(output_path):
    """
    Load existing persona_ids from the output file to skip already processed entries.
    """
    existing_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        persona_id = entry.get("persona_id")
                        if persona_id:
                            existing_ids.add(persona_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.warning(f"Could not load existing results from {output_path}: {e}")
    return existing_ids

def run_qwen_no_persona(
    questions,
    results_dir=RESULTS_DIR,
    output_basename=OUTPUT_NAME,
    model=MODEL,
    # sleep=1, # no need to sleep for the novita api
    max_retries=3
):
    """
    For each question, get the default response (no persona).
    Save to one jsonl file with persona_id = "<question_id>___gpt_oss_baseline"
    Skips questions that have already been processed.
    """
    output_path = os.path.join(results_dir, output_basename)
    os.makedirs(results_dir, exist_ok=True)

    # Load existing results to skip already processed questions
    existing_ids = load_existing_results(output_path)
    logging.info(f"Found {len(existing_ids)} existing entries in {output_path}")

    logging.info(f"==== Running baseline GPT-OSS-120B (no persona) responses for {len(questions)} questions ====")
    with open(output_path, "a", encoding="utf-8") as f_out:
        for q_idx, question in enumerate(questions):
            question_id = question["id"]
            input_text = question["input_text"]

            # Check if this question has already been processed
            persona_id = f"{question_id}___gpt_oss_baseline"
            if persona_id in existing_ids:
                logging.info(f"--> [{q_idx+1}/{len(questions)}] {question_id}: SKIP (already done)")
                continue

            logging.info(f"--> [{q_idx+1}/{len(questions)}] {question_id}: Processing")

            # Prompt is just the input + categories, no persona
            prompt = (
                "Given the following text post, classify it into one of the available categories:\n"
                "Available categories:\n"
                f"{', '.join(SCALE)}\n\n"
                "First, think step by step, explaining your reasoning inside <think>...</think> tags.\n"
                "Then, respond with a JSON object in this format (and no other text):\n"
                "{{\n"
                '  "label": "<one of the above categories>",\n'
                '  "rationale": ["word1", "word2", "..."]\n'
                "}}\n\n"
                "Input:\n"
                f"{input_text}\n"
            )

            response = None
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        extra_body={
                        "reasoning": {
                            "effort": "medium", 
                            "enabled": True
                            }
                        },
                    )
                    logging.info(f"    [✓] API call succeeded on attempt {attempt+1}")
                    break
                except Exception as e:
                    error_str = str(e)
                    if ("Service tier capacity exceeded" in error_str) or ("429" in error_str):
                        logging.warning(f"    [!] Capacity error, retrying ({attempt+1}/{max_retries})...")
                        time.sleep(10 * (attempt + 1))
                    else:
                        logging.error(f"    [ERROR] {question_id}: {e}")
                        break

            if response is None:
                logging.error(f"    [FAIL] Could not process {question_id} after {max_retries} retries.")
                continue

            raw_output = response.choices[0].message.content.strip()
            logging.debug(f"    [RAW OUTPUT]: {raw_output}")
            # print(response.choices[0])

            # Try to capture reasoning if provided
            reasoning_output = getattr(response.choices[0].message, "reasoning", None)
            # print(response.choices[0])
            if reasoning_output:
                logging.debug(f"    [REASONING OUTPUT]: {reasoning_output}")
                # You can choose to store this separately in your result dict
            else:
                logging.debug("    [NO REASONING OUTPUT FOUND]")

            # Parse the model's output
            try:
                match = re.search(r"\{[\s\S]*?\}", raw_output)
                if match:
                    parsed = json.loads(match.group(0))
                    label = parsed.get("label", "").strip()
                    rationale_words = parsed.get("rationale", [])
                else:
                    raise ValueError("No valid JSON found")
            except Exception as e:
                logging.error(f"    [!] JSON parse error: {e}")
                label = None
                rationale_words = []

            rationale_binary = compute_rationale_binary(input_text, rationale_words)

            # persona_id format: <question_id>___gpt_oss_baseline
            # (already defined above for skip check)

            result = {
                "persona_id": persona_id,
                "question_id": question_id,
                "label": label,
                "input_text": input_text,
                "rationale": rationale_words,
                "rationale_binary": rationale_binary,
                "raw_response": raw_output,
                "reasoning_output": reasoning_output 
            }

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            # time.sleep(sleep) # no need to sleep for the novita api

    logging.info(f"[✓] Baseline results saved to {output_path}")

# ========== MAIN ==========
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate baseline explanations with GPT-OSS-120B")
    parser.add_argument("--start", type=int, default=0, help="Start index for questions (0-based, inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index for questions (0-based, exclusive)")
    args = parser.parse_args()

    # Load questions
    with open(QUESTIONS_PATH, "r") as f:
        all_questions = [json.loads(line) for line in f]

    # Apply range selection
    start_idx = args.start
    end_idx = args.end if args.end is not None else len(all_questions)

    # Validate range
    if start_idx < 0 or start_idx >= len(all_questions):
        logging.error(f"Start index {start_idx} is out of range [0, {len(all_questions)-1}]")
        exit(1)
    if end_idx < start_idx or end_idx > len(all_questions):
        logging.error(f"End index {end_idx} is invalid (should be > {start_idx} and <= {len(all_questions)})")
        exit(1)

    questions = all_questions[start_idx:end_idx]
    logging.info(f"Processing questions {start_idx} to {end_idx-1} ({len(questions)} total questions)")

    if TEST_MODE:
        logging.warning("[!] Test mode ON — limiting to 2 questions.")
        questions = questions[:2]
    #     sleep = 1
    # else:
    #     sleep = 1

    run_qwen_no_persona(
        questions,
        results_dir=RESULTS_DIR,
        output_basename=OUTPUT_NAME,
        model=MODEL,
        # sleep=1,
        max_retries=3
    )
