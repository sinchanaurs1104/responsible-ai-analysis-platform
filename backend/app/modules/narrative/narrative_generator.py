"""
Layer 2 of the fairness-narrative design: rephrases an already-computed
FairnessFinding into plain language. Never computes the verdict itself
-- that's entirely FairnessFinding's job (Layer 1, insight_engine.py).

Validates the LLM's output before trusting it: every numeric token in
the response must already appear in the source JSON. If validation
fails, or the LLM call itself fails/is unavailable, falls back to the
deterministic template so the summary card always renders something.
"""

import re
from pathlib import Path

from app.schemas.fairness import FairnessFinding
from app.modules.narrative import llm_client
from app.modules.narrative.fallback_template import render_fallback_summary

PROMPT_PATH = Path(__file__).parent / "prompts" / "fairness_summary.txt"
NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def _extract_numbers(text: str) -> set[str]:
    """
    Normalizes matched numbers by stripping trailing zeros/decimal point
    so "32.70" and "32.7" are treated as equal -- the LLM may reformat
    precision without that counting as introducing a new number.
    """
    normalized = set()
    for match in NUMBER_PATTERN.findall(text):
        try:
            value = float(match)
            normalized.add(f"{value:g}")
        except ValueError:
            continue
    return normalized


def _validate_no_new_numbers(llm_output: str, finding_json: str) -> bool:
    output_numbers = _extract_numbers(llm_output)
    source_numbers = _extract_numbers(finding_json)
    # Every number in the output must be traceable to the source JSON.
    return output_numbers.issubset(source_numbers)


def generate_fairness_narrative(finding: FairnessFinding) -> str:
    finding_json = finding.model_dump_json(indent=2)
    prompt = PROMPT_PATH.read_text().format(finding_json=finding_json)

    llm_output = llm_client.generate_text(prompt)

    if llm_output is None:
        return render_fallback_summary(finding)

    if not _validate_no_new_numbers(llm_output, finding_json):
        # LLM introduced a number not grounded in the source data --
        # do not trust it, use the deterministic template instead.
        return render_fallback_summary(finding)

    return llm_output
