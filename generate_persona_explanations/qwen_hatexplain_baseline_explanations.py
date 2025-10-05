import json
import os
import re
import time
import logging
from openai import OpenAI

from rationale_binary_utils import compute_rationale_binary

# ========== CONFIGURATION ==========
API_KEY = "<YOUR_API_KEY>" 
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3-32b"
TEST_MODE = False  # Set to False for full run

# ==== LOGGING ====
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/generate_baseline_explanations_qwen.log", mode='a', encoding='utf-8')
    ]
)

# ========== QWEN CLIENT ==========
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# ========== DATA LOCATIONS ==========
QUESTIONS_PATH = "../datasets/personas_&_questions/hatexplain_500_hybrid.jsonl"
RESULTS_DIR = "../datasets/results_qwen3_32b_500_r1"
OUTPUT_NAME = "baseline_qwen3_32b.jsonl"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCALE = ["Hate speech", "Offensive language", "Normal"]

# ========== HELPER FUNCTIONS ==========

def run_qwen_no_persona(
    questions,
    results_dir=RESULTS_DIR,
    output_basename=OUTPUT_NAME,
    model=MODEL,
    # sleep=1, # no need to sleep for the novita api
    max_retries=3
):
    """
    For each question, get the default Qwen response (no persona).
    Save to one jsonl file with persona_id = "<question_id>___qwen_baseline"
    """
    output_path = os.path.join(results_dir, output_basename)
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f"==== Running baseline Qwen (no persona) responses for {len(questions)} questions ====")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for q_idx, question in enumerate(questions):
            question_id = question["id"]
            input_text = question["input_text"]
            logging.info(f"--> [{q_idx+1}/{len(questions)}] {question_id}")

            # Prompt is just the input + categories, no persona
            prompt = (
                "Given the following text post, classify it into one of the available categories:\n"
                "Available categories:\n"
                f"{', '.join(SCALE)}\n\n"
                "First, think step by step, explaining your reasoning inside <think>...</think> tags.\n"
                "Then, respond with a JSON object in this format (and no other text):\n"
                "{\n"
                '  "label": "<one of the above categories>",\n'
                '  "rationale": ["word1", "word2", "..."]\n'
                "}\n\n"
                "Input:\n"
                f"{input_text}\n"
            )

            response = None
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.6,
                        extra_body={
                        "top_k": 20,
                        "min_p": 0.0,
                        "top_p": 0.95,
                        "reasoning": {"enabled": True}
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

            # Try to capture reasoning if provided
            reasoning_output = getattr(response.choices[0].message, "reasoning", None)
            # print(response.choices[0])
            if reasoning_output:
                logging.debug(f"    [REASONING OUTPUT]: {reasoning_output}")
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

            # persona_id format: <question_id>___qwen_baseline
            persona_id = f"{question_id}___qwen_baseline"

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
    # Load questions
    with open(QUESTIONS_PATH, "r") as f:
        questions = [json.loads(line) for line in f]

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
