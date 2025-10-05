import json
import os
import re
import time
import logging
from typing import Any, Dict, List
from openai import OpenAI

from rationale_binary_utils import compute_rationale_binary

# ==== CONFIGURATION ====
API_KEY = "YOUR_API_KEY" 
BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "mistralai/mistral-medium-3.1"

# ==== LOGGING CONFIGURATION ====
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/mistral_sst_baseline.log", mode='a', encoding='utf-8'),
    ],
)


def load_qa_items(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


INSTRUCTION = (
    "Task: read the following sentence and choose exactly one of three labels.\n\n"
    "Positive: The sentence conveys information about the author’s positive evaluative sentiment.\n\n"
    "Negative: The sentence conveys information about the author’s negative evaluative sentiment.\n\n"
    "No sentiment: The sentence does not convey anything about the author’s positive or negative sentiment.\n\n"
    "Please specify what snippets of text, in the sentence, act as supporting evidence for your chosen label.\n"
    "First, think step by step inside <think>...</think> tags.\n\n"
    "Then, respond only with a JSON object in this format (and no other text):\n"
    "{{\n"
    '  "label": "<one of the above categories>",\n'
    '  "rationale": ["word1", "word2"],\n'
    '  "reasoning": "<your reasoning>"\n'
    "}}\n\n"
    "Sentence:\n{sentence}\n\n"
)


def make_prompt(item: Dict[str, Any]) -> str:
    sentence = str(item.get("sentence", "")).strip()
    return INSTRUCTION.format(sentence=sentence)



# Data files
qa_file = "../data/SST/BO_processed.json"
questions = load_qa_items(qa_file)

logging.info("[✓] Test mode OFF — processing all SST sentences.")

# Baseline results: single JSONL file, one entry per QID
results_dir = "../datasets/results_mistral_medium_sst_r3"
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, "baseline_mistral_sst.jsonl")

def parse_existing_jsonl(path: str) -> list:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def is_valid_label(label: str) -> bool:
    if label is None:
        return False
    lab = str(label).strip().lower()
    return lab in {"positive", "negative", "no sentiment"}


def upsert_jsonl_record(path: str, key_field: str, key_value: str, new_record: dict):
    records = parse_existing_jsonl(path)
    filtered = [r for r in records if r.get(key_field) != key_value]
    filtered.append(new_record)
    with open(path, "w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r) + "\n")

# Build a dict of existing records by QID and mark validity
existing_records = parse_existing_jsonl(output_file)
by_qid = {}
for rec in existing_records:
    qid = rec.get("question_id")
    if not qid:
        continue
    by_qid[qid] = rec

for q_idx, item in enumerate(questions):
    question_id = item.get("id") or item.get("QID") or f"q_{q_idx}"
    input_text = (item.get("sentence") or item.get("input_text") or "").strip()

    logging.info(f"=== [{q_idx+1}/{len(questions)}] Baseline for: {question_id} ===")

    existing = by_qid.get(question_id)
    if existing and is_valid_label(existing.get("model_answer")):
        logging.info(f"[=] {question_id}: valid answer already saved; skipping.")
        continue
    elif existing:
        logging.info(f"[~] {question_id}: found invalid/null answer; re-generating.")
    else:
        logging.info(f"[ ] {question_id}: no saved answer; generating.")

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
        label = (parsed.get("label") or "").strip()
        rationale_words = parsed.get("rationale", [])
        reasoning = parsed.get("reasoning", "")
        if not isinstance(rationale_words, list):
            rationale_words = [str(rationale_words)]
    except Exception as e:
        logging.error(f"    [!] JSON parse error: {e}")
        label = None
        rationale_words = []
        reasoning = None

    rationale_binary = compute_rationale_binary(input_text, rationale_words)

    baseline_pid = f"{question_id}___mistral_sst_baseline"
    result = {
        "persona_id": baseline_pid,
        "question_id": question_id,
        "input_text": input_text,
        "model_answer": label,
        "model_rationale": rationale_words,
        "rationale_binary": rationale_binary,
        "raw_response": raw_output,
        "reasoning_output": reasoning,
    }

    # Replace or append atomically for this question_id
    upsert_jsonl_record(output_file, "question_id", question_id, result)

    logging.info(f"[✓] Baseline saved to {output_file}")

# Final summary: how many QIDs still missing valid baseline outputs
def _valid_label(label: str) -> bool:
    if label is None:
        return False
    return str(label).strip().lower() in {"positive", "negative", "no sentiment"}

existing = parse_existing_jsonl(output_file)
valid_qids = {rec.get("question_id") for rec in existing if _valid_label(rec.get("model_answer"))}
missing = len(questions) - len(valid_qids)
if missing > 0:
    logging.warning(f"[SUMMARY] Missing baseline entries: {missing} QIDs without valid labels out of {len(questions)}.")
else:
    logging.info(f"[SUMMARY] All baseline entries present: {len(valid_qids)}/{len(questions)} QIDs valid.")
