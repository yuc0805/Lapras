"""Rewrites the reasoning instructions out of a prompt for No-CoT training (``--use_cot False``)."""

import re

# (pattern, replacement), applied in order.
_RULES: list[tuple[re.Pattern, str]] = [
    # Drop instruction bullets that ask for reasoning.
    (
        re.compile(
            r"^\s*-\s*(Reason carefully and methodically|Think step-by-step|"
            r"Write your reasoning|Write your rationale|Only reveal the correct class|"
            r"Do \*?\*?not\*?\*? mention any (class label|final answer))[^\n]*\n?",
            re.MULTILINE,
        ),
        "",
    ),
    # Rewrite the closing "please write your rationale" sentence (Sleep prompts).
    (
        re.compile(
            r"Please now write your rationale\.\s*"
            r"Make sure that your last word is the answer\.\s*"
            r"You MUST end your response with \"Answer:"
        ),
        "Provide your answer directly in the format: \"Answer:",
    ),
    # Rewrite the shorter closing variant (HAR / ECG prompts).
    (
        re.compile(
            r"Make sure that your last word is the answer\.\s*"
            r"You MUST end your response with \"Answer:"
        ),
        "Respond in the format: \"Answer:",
    ),
    # Collapse blank lines left behind by bullet removal.
    (re.compile(r"\n{3,}"), "\n\n"),
]


def strip_cot_instructions(text: str) -> str:
    """Apply every rewrite rule in order; return the cleaned text."""
    for pat, repl in _RULES:
        text = pat.sub(repl, text)
    return text.strip()
