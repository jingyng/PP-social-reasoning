import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from openai import OpenAI

from rationale_binary_utils import compute_rationale_binary

# ==== CONFIGURATION ====
client = OpenAI(
    api_key="<YOUR_API_KEY>",
    base_url="https://openrouter.ai/api/v1",
)

model = "qwen/qwen3-32b"
default_results_dir = "../datasets/results_qwen3_32b_sst_r3"

# ==== LOGGING CONFIGURATION ====
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/qwen_persona_sst.log", mode='a', encoding='utf-8'),
    ],
)


def load_personas(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            age = row.get("age")
            gender = row.get("gender") or row.get("Gender")
            ethnicity = row.get("ethnicity") or row.get("Ethnicity")
            persona_text = row.get("persona_text") or row.get("Persona_text") or row.get("persona")

            if not persona_text:
                if age and gender and ethnicity:
                    persona_text = f"A {age}-year-old {ethnicity} {gender}"
                else:
                    continue

            rows.append(
                {
                    "age": int(age) if str(age).isdigit() else age,
                    "gender": (gender or "").strip(),
                    "ethnicity": (ethnicity or "").strip(),
                    "persona_text": persona_text.strip(),
                }
            )
    return rows


def persona_id_from_parts(question_id: str, persona: Dict[str, Any]) -> str:
    parts = [str(persona.get("age", ""))]

    gender_map = {"male": "m", "female": "f"}
    eth_map = {"Caucasian": "w", "African American": "b", "Hispanic": "l"}

    gender = str(persona.get("gender", "")).strip().lower()
    ethnicity = str(persona.get("ethnicity", "")).strip()

    if gender in gender_map:
        parts.append(gender_map[gender])
    if ethnicity in eth_map:
        parts.append(eth_map[ethnicity])

    suffix = "_".join(parts) if parts else "na"
    return f"{question_id}__{suffix}"


def load_qa_items(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

INSTRUCTION = (
    "Step into the shoes of a real person who is {persona_desc}.\n\n"
    "Imagine you have lived your whole life shaped by this trait. How might it affect your thoughts, emotions, sensitivities, and judgments?\n\n"
    "Task: read the following sentence and choose from one of the three labels:\n\n"
    "Positive: The sentence conveys information about the author’s positive evaluative sentiment.\n\n"
    "Negative: The sentence conveys information about the author’s negative evaluative sentiment.\n\n"
    "No sentiment: The sentence does not convey anything about the author’s positive or negative sentiment.\n\n"
    "Please specify what snippets of text, in the sentence, you think acts as supporting evidence for your chosen label.\n"
    "First, *think out loud* as this persona, step by step, inside <think>...</think> tags. "
    "Reflect honestly about how this person's perspective influences what they notice and how they judge the text.\n\n"
    "Answer it exactly as this person would—letting their background, beliefs, and life experiences guide your reasoning.\n\n"
    "Then, respond only with a JSON object in this format (and no other text):\n"
    "{{\n"
    '  "label": "<one of the above categories>",\n'
    '  "rationale": ["word1", "word2"]\n'
    "}}\n\n"
    "Rules:\n"
    "- Stay fully in character; do not break persona or add explanations outside the JSON."
)


def make_prompt(item: Dict[str, Any], persona_desc: str) -> str:
    sentence = str(item.get("sentence", "")).strip()
    instruction_text = INSTRUCTION.format(persona_desc=persona_desc)
    return f"{instruction_text}\n\nSentence:\n{sentence}\n"



def _normalize_persona_id(pid: str) -> str:
    try:
        return re.sub(r"_{2,}(?=\d)", "___", pid)
    except Exception:
        return pid


def already_done_personas(output_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pid = json.loads(line).get("persona_id")
            except Exception:
                continue
            if pid:
                done.add(pid)
                done.add(_normalize_persona_id(pid))
    return done


personas_file = "../datasets/personas_&_questions/cose_personas.csv"
qa_file = "../data/SST/BO_processed.json"

personas = load_personas(personas_file)
all_items = load_qa_items(qa_file)

# Group by unique QID and keep first occurrence
qid_to_question = {}
for item in all_items:
    qid = item.get("id") or item.get("QID")
    if qid and qid not in qid_to_question:
        qid_to_question[qid] = item

questions = list(qid_to_question.values())

parser = argparse.ArgumentParser(description="Generate Qwen SST persona explanations, skipping existing entries.")
parser.add_argument(
    "--results-dir",
    default=default_results_dir,
    help="Directory where per-question JSONL outputs are stored",
)
parser.add_argument(
    "--start-question-id",
    type=int,
    default=0,
    help="Start processing from this question index (0-based)",
)
parser.add_argument(
    "--end-question-id",
    type=int,
    default=None,
    help="End processing at this question index (exclusive). If not specified, processes until the end.",
)
cli_args = parser.parse_args()

RESULTS_DIR = Path(cli_args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

start_question_id = cli_args.start_question_id
end_question_id = cli_args.end_question_id if cli_args.end_question_id is not None else len(questions)


def _iter_persona_runs(question_id: str, personas: List[Dict[str, Any]], done_persona_ids: Set[str]):
    total = len(personas)
    for p_idx, persona in enumerate(personas):
        persona_text = persona["persona_text"]
        raw_pid = persona_id_from_parts(question_id, persona)
        canonical_persona_id = _normalize_persona_id(raw_pid)

        log_prefix = f"[{p_idx + 1}/{total}] {persona_text} x {question_id}"

        if canonical_persona_id in done_persona_ids:
            logging.info(f"--> {log_prefix}: SKIP (already done)")
            continue

        logging.info(f"--> {log_prefix}: Processing")
        yield persona_text, canonical_persona_id, persona


for q_idx, question in enumerate(questions, start=1):
    if q_idx - 1 < start_question_id:
        continue
    if q_idx - 1 >= end_question_id:
        break

    question_id = question.get("id") or question.get("QID") or f"q_{q_idx}"
    input_text = (question.get("sentence") or question.get("input_text") or "").strip()

    output_path = RESULTS_DIR / f"{question_id}.jsonl"
    logging.info(f"=== [{q_idx}/{len(questions)}] Running analysis for question id: {question_id} ===")

    done_persona_ids = already_done_personas(output_path)

    with output_path.open("a", encoding="utf-8") as f_out:
        for persona_text, canonical_persona_id, persona in _iter_persona_runs(question_id, personas, done_persona_ids):
            prompt = make_prompt(question, persona_text)
            start_time = time.time()

            response = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.6,
                        extra_body={
                            "top_k": 20,
                            "min_p": 0.0,
                            "top_p": 0.95,
                            "reasoning": {"enabled": True},
                        },
                    )
                    logging.info(f"    [✓] API call succeeded on attempt {attempt + 1}")
                    break
                except Exception as e:
                    error_str = str(e)
                    if "Service tier capacity exceeded" in error_str or "429" in error_str:
                        logging.warning(f"    [!] Capacity error, retrying ({attempt + 1}/3)...")
                        time.sleep(10 * (attempt + 1))
                    else:
                        logging.error(f"    [ERROR] {persona_text} x {question_id}: {e}")
                        break

            if response is None:
                logging.error(f"    [FAIL] Could not process {persona_text} x {question_id} after retries.")
                continue

            raw_output = response.choices[0].message.content.strip()
            logging.debug(f"    [RAW OUTPUT]: {raw_output}")

            reasoning_output = getattr(response.choices[0].message, "reasoning", None)

            try:
                match = re.search(r"\{[\s\S]*?\}", raw_output)
                if not match:
                    raise ValueError("No JSON object found in output")
                parsed = json.loads(match.group(0))

                if "label" in parsed:
                    label = parsed.get("label", "").strip()
                    answer_index = parsed.get("answer_index")
                elif "answer" in parsed:
                    label = parsed.get("answer", "").strip()
                    answer_index = parsed.get("answer_index")
                else:
                    label = parsed.get("label_agreement", "").strip()
                    answer_index = parsed.get("answer_index")

                rationale_words = parsed.get("rationale", [])
                if not isinstance(rationale_words, list):
                    rationale_words = [str(rationale_words)]
            except Exception as e:
                logging.error(f"    [!] JSON parse error: {e}")
                label = None
                answer_index = None
                rationale_words = []

            rationale_binary = compute_rationale_binary(input_text, rationale_words)

            result = {
                "persona_id": canonical_persona_id,
                "persona_text": persona_text,
                "question_id": question_id,
                "input_text": input_text,
                "model_answer": label,
                "model_answer_index": answer_index,
                "model_rationale": rationale_words,
                "rationale_binary": rationale_binary,
                "raw_response": raw_output,
                "reasoning_output": reasoning_output,
            }

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

            elapsed = time.time() - start_time
            done_persona_ids.add(canonical_persona_id)
            logging.info(f"    [✓] Done: {persona_text} x {question_id} in {elapsed:.2f} seconds")

    logging.info(f"[✓] Results saved to {output_path}")

logging.info("=== Done processing all questions ===")
