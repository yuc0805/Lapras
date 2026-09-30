"""Answer parsing and metrics (accuracy, macro-F1) for time-series QA."""

import re

from sklearn.metrics import f1_score


def extract_gt_label(text: str) -> str | None:
    """Extract ground-truth label from 'Answer: <label>' at end of output."""
    # The last match: a CoT may contain earlier "answer:" phrases.
    matches = re.findall(r"Answer:\s*(.+)", text, re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip(".").strip('"').strip("'")
    return None


def build_label_set(samples: list[dict]) -> set[str]:
    """Auto-discover the set of valid labels from ground-truth outputs."""
    labels = set()
    for s in samples:
        label = extract_gt_label(s["output"])
        if label:
            labels.add(label)
    return labels


_TRIM_CHARS = " \t\n\r.,;:!?\"'`*"
# Leading MCQ option markers: "A) ", "(a) ", "[B] ", "1. ", "2: "
_OPT_PREFIX_RE = re.compile(r"^[\(\[]?[A-Za-z0-9][\)\]\.\:]\s+")
_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def _match_label(s: str, valid_labels: set[str]) -> str | None:
    if not s:
        return None
    if s in valid_labels:
        return s
    sl = s.lower()
    for label in valid_labels:
        if sl == label.lower():
            return label
    return None


def _strong_norm(s: str) -> str:
    """Lowercase and strip one leading option marker and one leading article."""
    s = s.strip().strip(_TRIM_CHARS)
    s = _OPT_PREFIX_RE.sub("", s, count=1).strip()
    s = _ARTICLE_RE.sub("", s, count=1).strip()
    return s.lower()


def _build_strong_norm_map(valid_labels: set[str]) -> dict[str, str]:
    """Map normalized form -> label, dropping forms shared by several labels."""
    out: dict[str, str] = {}
    bad: set[str] = set()
    for label in valid_labels:
        key = _strong_norm(label)
        if not key:
            continue
        if key in out and out[key] != label:
            bad.add(key)
        else:
            out[key] = label
    for k in bad:
        out.pop(k, None)
    return out


def extract_pred_label(text: str, valid_labels: set[str]) -> str | None:
    """Label after the last ``Answer:``: exact, then normalized match, never a substring match."""
    strong_map = _build_strong_norm_map(valid_labels)

    def _resolve(candidate: str) -> str | None:
        hit = _match_label(candidate, valid_labels)
        if hit is not None:
            return hit
        key = _strong_norm(candidate)
        if key and key in strong_map:
            return strong_map[key]
        return None

    matches = re.findall(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
    if matches:
        candidate = matches[-1].strip().strip(_TRIM_CHARS)
        hit = _resolve(candidate)
        if hit is not None:
            return hit
        # Fall back to the last whole token of the answer line.
        toks = candidate.split()
        if toks:
            last = toks[-1].strip(_TRIM_CHARS)
            hit = _match_label(last, valid_labels)
            if hit is not None:
                return hit
        return None
    # Without an "Answer:" line, only a final token that is itself a label counts.
    tail_tokens = text.strip().split()
    if tail_tokens:
        last = tail_tokens[-1].strip(_TRIM_CHARS)
        return _match_label(last, valid_labels)
    return None


_GT_LETTER_RE   = re.compile(r"\(?([A-E])[\)\.,:]")          # leading "A," / "(A)" / "A."
_ANS_LETTER_RE  = re.compile(r"Answer\s*:?\s*\(?([A-E])\b", re.IGNORECASE)
_BARE_LETTER_RE = re.compile(r"\b([A-E])\b")


def extract_gt_letter(text: str) -> str | None:
    """Leading option letter of the gold answer ('A, ...' / '(A) ...' -> 'A')."""
    label = extract_gt_label(text)
    if label is None:
        return None
    m = _GT_LETTER_RE.match(label.strip())
    return m.group(1).upper() if m else None


def build_letter_set(samples: list[dict]) -> set[str]:
    """Auto-discover the set of valid option letters from ground-truth outputs."""
    return {l for s in samples if (l := extract_gt_letter(s["output"])) is not None}


def extract_pred_letter(text: str, valid_letters: set[str]) -> str | None:
    """Option letter after 'Answer:', else the first standalone valid letter, else None."""
    m = _ANS_LETTER_RE.search(text)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()
    for m in _BARE_LETTER_RE.finditer(text):
        if m.group(1).upper() in valid_letters:
            return m.group(1).upper()
    return None


def is_mcq_letter_set(valid_labels: set[str]) -> bool:
    """Whether the gold labels are option letters: >= 60% letter-prefixed, with >= 2 distinct letters."""
    if not valid_labels:
        return False
    hits = [m.group(1).upper() for l in valid_labels
            if (m := _GT_LETTER_RE.match(l.strip()))]
    return len(hits) >= 0.6 * len(valid_labels) and len(set(hits)) >= 2


def prepare_label_space(samples: list[dict]):
    """Return the label set, whether it is multiple choice, and its option letters."""
    valid_labels = build_label_set(samples)
    mcq = is_mcq_letter_set(valid_labels)
    valid_letters = build_letter_set(samples) if mcq else set()
    return valid_labels, mcq, valid_letters


def score_sample(pred_text: str, gt_output: str, valid_labels: set[str],
                 mcq: bool, valid_letters: set[str]):
    """(gold, predicted) key of one sample: option letters for multiple choice, labels otherwise."""
    if mcq:
        return extract_gt_letter(gt_output), extract_pred_letter(pred_text, valid_letters)
    return extract_gt_label(gt_output), extract_pred_label(pred_text, valid_labels)


def compute_metrics(gt_labels: list[str], pred_labels: list) -> dict:
    """Accuracy and macro-F1 (%) over scored samples. ``None`` predictions count as ``<none>``."""
    gold = [str(g) for g in gt_labels]
    pred = [str(p) if p is not None and str(p).strip() else "<none>" for p in pred_labels]
    labels = sorted(set(gold))
    n = len(gold)
    return {
        "n": n,
        "correct": sum(g == p for g, p in zip(gold, pred)),
        "accuracy": 100.0 * sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0,
        "macro_f1": 100.0 * f1_score(gold, pred, labels=labels, average="macro", zero_division=0) if n else 0.0,
        "n_unparsed": sum(p == "<none>" for p in pred),
    }
