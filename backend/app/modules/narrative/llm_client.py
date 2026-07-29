"""
Thin wrapper around the Anthropic API. Isolated in its own file so it's
the only place in the codebase that touches the SDK -- swapping models
or providers later is a one-file change.

Returns None on ANY failure (missing key, network error, SDK error)
rather than raising -- narrative_generator.py treats None as "use the
fallback template," per the SDD's "narration is additive polish, never
a hard dependency" decision.
"""

import os

DEFAULT_MODEL = os.environ.get("NARRATIVE_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 400


def generate_text(prompt: str) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "".join(text_blocks).strip() or None
    except Exception:  # noqa: BLE001 -- any failure here just falls back to the template
        return None
