import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Set

from openai import OpenAI

from rationale_binary_utils import compute_rationale_binary

# ==== CONFIGURATION ====
API_KEY = "<YOUR_API_KEY>" 
BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "mistralai/mistral-medium-3.1"

DEFAULT_RESULTS_DIR = "../datasets/results_mistral_medium_500_r2"

# ==== LOGGING CONFIGURATION ====
os.makedirs("../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../logs/mistral_persona_prompting.log", mode='a', encoding='utf-8'),
    ],
)

SCALE = ["Hate speech", "Offensive language", "Normal"]
INSTRUCTION = (
    "Step into the shoes of a real person who {persona_desc}.\n\n"
    "Imagine you have lived your whole life shaped by this trait. How might it affect your thoughts, emotions, sensitivities, and judgments?\n\n"
    "Given the following text post, analyze and classify it exactly as this person would—letting their background, beliefs, and life experiences guide your reasoning.\n\n"
    "Available categories for classification:\n"
    "{scale}\n\n"
    "First, *think out loud* as this persona, step by step, inside <think>...</think> tags. "
    "Reflect honestly about how this person's perspective influences what they notice and how they judge the text.\n\n"
    "Then, respond only with a JSON object in this format (and no other text):\n"
    "{{\n"
    '  "label": "<one of the above categories>",\n'
    '  "rationale": ["word1", "word2", "..."]\n'
    '  "reasoning": "<your reasoning>"\n'
    "}}\n\n"
    "- Stay fully in character; do not break persona or add explanations outside the JSON."
)

ATTRIBUTE_RENDER = {
    "age": lambda v: f"is {v}-year-old",
    "gender": lambda v: f"is {v}",
    "race": lambda v: f"is {v}",
    "education": lambda v: f"has {v}",
    "religion": lambda v: f"is a {v}",
    "political_view": lambda v: f"has {v} political views",
    "loneliness": lambda v: f"is {v}",
}

ATTRIBUTE_ID_MAP = {
    "gender": {"male": "m", "female": "f"},
    "education": {
        "no formal education": "nfe",
        "high school education": "hs",
        "higher education": "he",
    },
    "race": {"white": "w", "black": "b", "asian": "a"},
    "religion": {
        "Christian": "chr",
        "Muslim": "mus",
        "Jewish": "jew",
        "Atheist": "ath",
        "Hindu": "hin",
    },
    "political_view": {"left-wing": "l", "right-wing": "r", "centrist": "c"},
    "loneliness": {"not lonely": "nl", "somewhat lonely": "sl"},
}


def load_personas(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def persona_description(persona: Dict[str, str]) -> str:
    parts: List[str] = []
    for key, renderer in ATTRIBUTE_RENDER.items():
        value = persona.get(key)
        if value:
            parts.append(renderer(str(value)))
    if not parts:
        return "has an unspecified background"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def generate_persona_id(question_id: str, persona: Dict[str, str]) -> str:
    parts: List[str] = []
    age = persona.get("age")
    if age is not None:
        parts.append(str(age))
    for attr, mapping in ATTRIBUTE_ID_MAP.items():
        value = persona.get(attr)
        if value in mapping:
            parts.append(mapping[value])
    suffix = "_".join(parts) if parts else "na"
    prefix = "___" if suffix and suffix[0].isdigit() else "__"
    return f"{question_id}{prefix}{suffix}"


def make_prompt(persona: Dict[str, str], input_text: str) -> str:
    return INSTRUCTION.format(
        persona_desc=persona_description(persona),
        scale=", ".join(SCALE),
    ) + f"\n\nInput:\n{input_text}\n"



def _normalize_persona_id(pid: str) -> str:
    try:
        match = re.match(r"^(.*?)(__+)(.+)$", pid)
        if not match:
            return pid
        question, _, suffix = match.groups()
        prefix = "___" if suffix and suffix[0].isdigit() else "__"
        return f"{question}{prefix}{suffix}"
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


personas = load_personas("../datasets/personas_&_questions/personas.jsonl")
questions = load_questions("../datasets/personas_&_questions/hatexplain_500_hybrid.jsonl")

parser = argparse.ArgumentParser(description="Generate Mistral HateXplain persona explanations, skipping existing entries.")
parser.add_argument(
    "--results-dir",
    default=DEFAULT_RESULTS_DIR,
    help="Directory where per-question JSONL outputs are stored",
)
parser.add_argument(
    "--start-question-id",
    type=int,
    default=0,
    help="Starting question index (0-based, inclusive)",
)
parser.add_argument(
    "--end-question-id",
    type=int,
    default=None,
    help="Ending question index (0-based, exclusive)",
)
cli_args = parser.parse_args()

RESULTS_DIR = Path(cli_args.results_dir)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Filter questions based on start and end indices
start_idx = cli_args.start_question_id
end_idx = cli_args.end_question_id if cli_args.end_question_id is not None else len(questions)
filtered_questions = questions[start_idx:end_idx]
logging.info(f"Processing questions {start_idx} to {end_idx - 1} (total: {len(filtered_questions)} questions)")

for q_idx, question in enumerate(filtered_questions, start=start_idx + 1):
    question_id = question.get("id") or question.get("post_id") or f"q_{q_idx}"
    input_text = question.get("input_text", "").strip()
    output_path = RESULTS_DIR / f"{question_id}.jsonl"
    logging.info(f"=== [{q_idx - start_idx}/{len(filtered_questions)}] Running analysis for question id: {question_id} ===")

    done_persona_ids = already_done_personas(output_path)

    with output_path.open("a", encoding="utf-8") as f_out:
        total = len(personas)
        for idx, persona in enumerate(personas):
            persona_id = generate_persona_id(question_id, persona)
            canonical_persona_id = _normalize_persona_id(persona_id)
            persona_label = persona_description(persona)
            log_prefix = f"[{idx + 1}/{total}] {persona_label} x {question_id}"

            if persona_id in done_persona_ids or canonical_persona_id in done_persona_ids:
                logging.info(f"--> {log_prefix}: SKIP (already done)")
                continue

            logging.info(f"--> {log_prefix}: Processing")

            prompt = make_prompt(persona, input_text)
            start_time = time.time()

            response = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        response_format={"type": "json_object"},
                    )
                    logging.info(f"    [✓] API call succeeded on attempt {attempt + 1}")
                    break
                except Exception as e:
                    error_str = str(e)
                    if "Service tier capacity exceeded" in error_str or "429" in error_str:
                        logging.warning(f"    [!] Capacity error, retrying ({attempt + 1}/3)...")
                        time.sleep(10 * (attempt + 1))
                    else:
                        logging.error(f"    [ERROR] {persona_label} x {question_id}: {e}")
                        break

            if response is None:
                logging.error(f"    [FAIL] Could not process {persona_label} x {question_id} after retries.")
                continue

            raw_output = response.choices[0].message.content.strip()
            logging.debug(f"    [RAW OUTPUT]: {raw_output}")

            try:
                match = re.search(r"\{[\s\S]*?\}", raw_output)
                if not match:
                    raise ValueError("No JSON object found in output")
                parsed = json.loads(match.group(0))
                label = parsed.get("label", "").strip()
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

            result = {
                "persona_id": persona_id,
                "question_id": question_id,
                "input_text": input_text,
                "label": label,
                "rationale": rationale_words,
                "rationale_binary": rationale_binary,
                "raw_response": raw_output,
                "reasoning_output": reasoning,
            }

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

            elapsed = time.time() - start_time
            done_persona_ids.update({persona_id, canonical_persona_id})
            logging.info(f"    [✓] Done: {persona_label} x {question_id} in {elapsed:.2f} seconds")

    logging.info(f"[✓] Results saved to {output_path}")

logging.info("=== Done processing all questions ===")
