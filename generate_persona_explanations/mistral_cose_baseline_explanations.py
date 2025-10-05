import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Any, Dict, List
from openai import OpenAI

from rationale_binary_utils import compute_rationale_binary

# ==== CONFIGURATION ====
API_KEY = "<YOUR_API_KEY>" 
BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "mistralai/mistral-medium-3.1"


# ==== LOGGING CONFIGURATION ====
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/mistral_cose_baseline.log", mode='a', encoding='utf-8'),
    ],
)


def load_qa_items(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


INSTRUCTION = (
    "Task: first select the answer you think is most correct and sensible; then mark relevant words in the question that justify your choice.\n"
    "First, think step by step inside <think>...</think> tags.\n\n"
    "Then, respond only with a JSON object in this format (and no other text):\n"
    "{{\n"
    '  "answer": "<exact option text>",\n'
    '  "answer_index": <0-based index>,\n'
    '  "rationale": ["keyword1", "keyword2"],\n'
    '  "reasoning": "<your reasoning>"\n'
    "}}\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
    "Rules:\n"
    "- The value for \"answer\" must be copied verbatim from the options list.\n"
    "- \"answer_index\" must match the position (0-based) in the options list.\n"
    "- For each word in the question, if you think removing it will decrease your confidence toward your chosen label, you should mark it.\n"
)


def format_options(opts: List[str]) -> str:
    return "\n".join(f"{i}. {opt}" for i, opt in enumerate(opts or []))


def make_prompt(item: Dict[str, Any]) -> str:
    question = str(item.get("sentence", "")).strip()
    options = item.get("query_list")
    if not options and item.get("query"):
        options = [s.strip() for s in str(item["query"]).split("[sep]")]
    return INSTRUCTION.format(question=question, options=format_options(options))



def count_jsonl_lines(p: str) -> int:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


# Data files
qa_file = "../data/cose/BO_processed_with_queries.json"
questions = load_qa_items(qa_file)

logging.info("[✓] Test mode OFF — processing all questions.")

# Baseline results: single JSONL file, one entry per QID
results_dir = "../datasets/results_mistral_medium_cose_r1"
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, "baseline_mistral_cose.jsonl")

# Build a set of question_ids already written (idempotent appends)
done_qids = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid:
                    done_qids.add(qid)
            except Exception:
                continue

for q_idx, item in enumerate(questions):
    question_id = item.get("id") or item.get("QID") or f"q_{q_idx}"
    input_text = (item.get("sentence") or item.get("input_text") or "").strip()

    logging.info(f"=== [{q_idx+1}/{len(questions)}] Baseline for: {question_id} ===")

    # Skip if this QID already exists in the combined baseline file
    if question_id in done_qids:
        logging.info(f"[=] {question_id}: already present; skipping.")
        continue

    prompt = make_prompt(item)

    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
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

    try:
        match = re.search(r"\{[\s\S]*?\}", raw_output)
        if not match:
            raise ValueError("No JSON object found in output")
        parsed = json.loads(match.group(0))
        answer = (parsed.get("answer") or "").strip()
        answer_index = parsed.get("answer_index", None)
        rationale_words = parsed.get("rationale", [])
        reasoning = parsed.get("reasoning", "")
    except Exception as e:
        logging.error(f"    [!] JSON parse error: {e}")
        answer = None
        answer_index = None
        rationale_words = []
        reasoning = None

    rationale_binary = compute_rationale_binary(input_text, rationale_words)

    baseline_pid = f"{question_id}___mistral_cose_baseline"
    result = {
        "persona_id": baseline_pid,
        "question_id": question_id,
        "input_text": input_text,
        "model_answer": answer,
        "model_answer_index": answer_index,
        "model_rationale": rationale_words,
        "rationale_binary": rationale_binary,
        "raw_response": raw_output,
        "reasoning_output": reasoning,
    }

    # Append a single line per question to the combined baseline file
    with open(output_file, "a", encoding="utf-8") as f_out:
        f_out.write(json.dumps(result) + "\n")
        f_out.flush()

    logging.info(f"[✓] Baseline saved to {output_file}")
