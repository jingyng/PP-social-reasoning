import json
import os
import re
import time
import csv
import logging
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set
from openai import OpenAI

from rationale_binary_utils import compute_rationale_binary

# ==== CONFIGURATION ====
client = OpenAI(
    api_key="<YOUR_API_KEY>",
    base_url="https://openrouter.ai/api/v1"
)

model = "qwen/qwen3-32b"



# ==== LOGGING CONFIGURATION ====
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("../logs/qwen_persona_cose.log", mode='a', encoding='utf-8')  # Log to logs folder
    ]
)

def load_personas(path: str) -> List[Dict[str, Any]]:
    """
    Reads a CSV with headers: age, gender, ethnicity, persona_text
    Returns a list of dicts with keys: age (int), gender (str), ethnicity (str), persona_text (str)
    """
    rows = []
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
                # Fallback: build persona_text if missing
                if age and gender and ethnicity:
                    persona_text = f"A {age}-year-old {ethnicity} {gender}"
                else:
                    # skip malformed line
                    continue

            rows.append({
                "age": int(age) if str(age).isdigit() else age,
                "gender": (gender or "").strip(),
                "ethnicity": (ethnicity or "").strip(),
                "persona_text": persona_text.strip(),
            })
    return rows

def persona_id_from_parts(question_id: str, persona: Dict[str, Any]) -> str:
    """
    Build persona_id like: question_id__25_m_b
    - age as number
    - gender mapped to 'm' or 'f'
    - ethnicity mapped to 'w' (Caucasian), 'b' (African American), 'l' (Hispanic)
    """
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

def _normalize_persona_id(pid: str) -> str:
    """Normalize persona_id to use exactly three underscores before age digits.
    - Any run of 2 or more underscores immediately before the age digits becomes exactly 3.
    """
    try:
        return re.sub(r"_{2,}(?=\d)", "___", pid)
    except Exception:
        return pid


def load_qa_items(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

INSTRUCTION = (
    "Step into the shoes of a real person who {persona_desc}.\n\n"
    "Imagine you have lived your whole life shaped by this trait. How might it affect your thoughts, emotions, sensitivities, and judgments?\n\n"
    "Task: first select the answer you think is most correct and sensible; then mark relevant words in the question that justifies your choice.\n"
    "First, *think out loud* as this persona, step by step, inside <think>...</think> tags. "
    "Reflect honestly about how this person's perspective influences what they notice and how they judge the text.\n\n"
    "Answer it exactly as this person would—letting their background, beliefs, and life experiences guide your reasoning.\n\n"
    "Then, respond only with a JSON object in this format (and no other text):\n"
    '{{"answer": "<exact option text>", "answer_index": <0-based index>, '
    '"rationale": ["keyword1", "keyword2"]}}\n\n'
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
    "Rules:\n"
    "- The value for \"answer\" must be copied verbatim from the options list.\n"
    "- \"answer_index\" must match the position (0-based) in the options list.\n"
    "- For each word in the question, if you think that removing it will decrease your confidence toward your chosen label, you should mark it.\n"
    "- Stay fully in character; do not break persona or add explanations outside the JSON."
)

def format_options(opts: List[str]) -> str:
    # Show with indices to make parsing unambiguous
    return "\n".join(f"{i}. {opt}" for i, opt in enumerate(opts))

def make_prompt(item: Dict[str, Any], persona_desc: str) -> str:
    # Use `sentence` as the question, prefer `query_list`, fallback to parsing `query`
    question = str(item.get("sentence", "")).strip()
    options = item.get("query_list")
    if not options and item.get("query"):
        # fallback: parse "a [sep] b [sep] c"
        options = [s.strip() for s in str(item["query"]).split("[sep]")]

    options = options or []  # ensure list
    return INSTRUCTION.format(
        persona_desc=persona_desc,
        question=question,
        options=format_options(options),
    )

def persona_id_from_text(question_id: str, persona_text: str) -> str:
    """Stable short ID; avoids underscore hacks."""
    h = hashlib.sha1(persona_text.encode("utf-8")).hexdigest()[:8]
    return f"{question_id}__{h}"

def already_done_personas(output_path):
    done = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = rec.get("persona_id")
                if pid:
                    done.add(pid)
                    done.add(_normalize_persona_id(pid))
            except Exception:
                continue
    return done

# Personas file: switch to a TXT with one persona per line if possible.
personas_file = "../datasets/personas_&_questions/cose_personas.csv"  # or ".../personas.txt"
qa_file = "../data/cose/BO_processed_with_queries.json"

personas = load_personas(personas_file)
questions = load_qa_items(qa_file)

parser = argparse.ArgumentParser(description="Generate Qwen COSE persona explanations, skipping existing entries.")
parser.add_argument(
    "--results-dir",
    default="../datasets/results_qwen3_32b_cose_r3",
    help="Directory where per-question JSONL outputs are stored",
)
cli_args = parser.parse_args()

RESULTS_DIR = Path(cli_args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    question_id = question.get("id") or question.get("QID") or f"q_{q_idx}"
    # For rationale mask, prefer the exact question text used in prompt
    input_text = (question.get("sentence") or question.get("input_text") or "").strip()

    output_path = RESULTS_DIR / f"{question_id}.jsonl"
    logging.info(f"=== [{q_idx}/{len(questions)}] Running analysis for question id: {question_id} ===")

    done_persona_ids = already_done_personas(output_path)

    with output_path.open("a", encoding="utf-8") as f_out:
        for persona_text, canonical_persona_id, persona in _iter_persona_runs(
            question_id, personas, done_persona_ids
        ):
            prompt = make_prompt(question, persona_text)
            start_time = time.time()

            max_retries = 3
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
                        logging.error(f"    [ERROR] {persona_text} x {question_id}: {e}")
                        break

            if response is None:
                logging.error(f"    [FAIL] Could not process {persona_text} x {question_id} after {max_retries} retries.")
                continue

            raw_output = response.choices[0].message.content.strip()
            logging.debug(f"    [RAW OUTPUT]: {raw_output}")

            # Some providers include a separate 'reasoning' field
            reasoning_output = getattr(response.choices[0].message, "reasoning", None)
            if reasoning_output:
                logging.debug(f"    [REASONING OUTPUT]: {reasoning_output}")
            else:
                logging.debug("    [NO REASONING OUTPUT FOUND]")

            # Parse model JSON — accept either {'answer', 'answer_index'} or {'label_agreement'}
            try:
                match = re.search(r"\{[\s\S]*?\}", raw_output)
                if not match:
                    raise ValueError("No JSON object found in output")
                parsed = json.loads(match.group(0))

                # Normalize fields
                if "answer" in parsed:
                    label = parsed.get("answer", "").strip()
                    answer_index = parsed.get("answer_index", None)
                else:
                    # Backward-compat with older prompts
                    label = parsed.get("label_agreement", "").strip()
                    answer_index = parsed.get("answer_index", None)

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
