"""Utilities for computing rationale token spans across COSE/SST style datasets."""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Optional, Sequence, Tuple

EDGE_PUNCTUATION = set(string.punctuation) | {"“", "”", "‘", "’", "…", "—", "–", "·", "•"}


def _normalize_token(token: str) -> str:
    lowered = token.lower().strip()
    if "'" in lowered:
        lowered = lowered.replace("'", "")
    start = 0
    end = len(lowered)
    while start < end and lowered[start] in EDGE_PUNCTUATION:
        start += 1
    while end > start and lowered[end - 1] in EDGE_PUNCTUATION:
        end -= 1
    return lowered[start:end]


def _split_phrase(phrase: str) -> List[str]:
    tokens: List[str] = []
    for part in phrase.split():
        norm = _normalize_token(part)
        if norm:
            tokens.append(norm)
    return tokens


def _find_full_span(normalized_tokens: Sequence[str], phrase_tokens: Sequence[str]) -> Optional[Tuple[int, int]]:
    if not phrase_tokens:
        return None

    best: Optional[Tuple[int, int]] = None
    token_count = len(normalized_tokens)

    for start in range(token_count):
        idx = start
        success = True

        for target in phrase_tokens:
            if idx >= token_count:
                success = False
                break

            concat = ""
            part_idx = idx
            while part_idx < token_count and concat != target:
                piece = normalized_tokens[part_idx]
                if not piece:
                    part_idx += 1
                    continue

                candidate = concat + piece
                if target.startswith(candidate):
                    concat = candidate
                    part_idx += 1
                else:
                    success = False
                    break

            if not success or concat != target:
                success = False
                break

            idx = part_idx

        if not success:
            continue

        span_length = idx - start
        if span_length <= 0:
            continue

        if best is None or span_length > best[1]:
            best = (start, span_length)

    return best


def _find_best_span(normalized_tokens: Sequence[str], phrase_tokens: Sequence[str]) -> Optional[Tuple[int, int]]:
    candidate = _find_full_span(normalized_tokens, phrase_tokens)
    if candidate is not None:
        return candidate

    for size in range(len(phrase_tokens) - 1, 0, -1):
        for offset in range(len(phrase_tokens) - size + 1):
            sub_tokens = phrase_tokens[offset : offset + size]
            candidate = _find_full_span(normalized_tokens, sub_tokens)
            if candidate is not None:
                return candidate

    return None


def compute_rationale_mask(tokens: Sequence[str], rationale_phrases: Iterable[str]) -> Tuple[List[int], List[str]]:
    mask = [0] * len(tokens)
    unmatched_phrases: List[str] = []
    normalized_tokens = [_normalize_token(tok) for tok in tokens]

    for phrase in rationale_phrases or []:
        segments = re.split(r"(?:\.\.\.|…)", phrase)
        matched = False
        for segment in segments:
            phrase_tokens = _split_phrase(segment)
            if not phrase_tokens:
                continue
            span = _find_best_span(normalized_tokens, phrase_tokens)
            if span is None:
                continue
            start, length = span
            for idx in range(start, start + length):
                if 0 <= idx < len(mask):
                    mask[idx] = 1
            matched = True
        if not matched:
            unmatched_phrases.append(phrase)

    return mask, unmatched_phrases


def compute_rationale_binary(input_text: str, rationale_phrases: Iterable[str]) -> List[int]:
    tokens = input_text.split()
    mask, _ = compute_rationale_mask(tokens, rationale_phrases)
    return mask
