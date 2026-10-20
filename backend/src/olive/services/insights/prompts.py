"""Editable prompts and tuning config for Olive Insights extraction.

Everything the model is told lives here as data, never as code logic: the behavioral rubric,
the named reference frameworks, and the ordinal vocabulary the model answers in. Defaults are
seeded to ``<configDir>/insights/prompts.json`` and ``config.json`` on first use; any key the
user sets in those files overrides the default for that key only, so the rubric can be tuned
without a code change. The prompt text deliberately carries no keyword lists and no numeric
thresholds; the numbers behind ordinal words live in ``DEFAULT_CONFIG["dimensionScale"]``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...logging_db import logger

SYSTEM_PROMPT = """\
You are a careful analyst of workplace and interpersonal communication. You read a diarized \
meeting transcript and describe how each speaker behaved, using established communication \
frameworks as a reference lens. You describe behavior that is visible in the words on the page; \
you never diagnose a person, infer a mental-health condition, or judge character. You only \
report what the transcript supports, and every notable event you report is tied to a short \
verbatim quote copied exactly from a cited turn. When nothing notable happened, say so by \
returning empty lists. You reply with a single JSON object and nothing else."""

FRAMEWORKS = """\
Use these frameworks as your reference lens. Name the framework (and the specific marker) in the \
"framework" field of any moment it applies to, for example "gottman:contempt" or "nvc:demand".

Gottman conflict markers:
- criticism: attacking a person's character or global worth rather than raising a specific behavior or need.
- contempt: expressing superiority or disdain, such as mockery, sarcasm meant to demean, name-calling, or dismissive put-downs.
- defensiveness: deflecting responsibility, counter-blaming, or playing the victim in response to a concern.
- stonewalling: withdrawing from the exchange, giving minimal or evasive replies, or refusing to engage.
- repair attempts: a clear effort to de-escalate or reconnect, such as an apology, humor that eases tension, acknowledging the other person's point, or asking to reset. Note whether the repair was accepted.
- bids for connection: an invitation for attention or support, and whether it was met or ignored.

Nonviolent Communication (NVC):
- observation versus evaluation: stating what happened versus judging the person.
- feelings and needs: naming a feeling or need versus blaming.
- request versus demand: a request leaves room for "no"; a demand carries blame, threat, or pressure.

Communication stance, per speaker:
- assertive: states needs and views directly while respecting the other person.
- aggressive: pushes needs at the other person's expense through blame, intimidation, or domination.
- passive: avoids stating needs or views, defers, or goes silent.
- passive-aggressive: indirect hostility such as veiled digs, sarcasm, or compliance with resentment.

Also notice active listening and validation, accountability versus blame, interruption and \
dominance of the floor, and significant interpersonal events such as accusation, termination or \
firing, threat or ultimatum, apology, praise or recognition, and escalation or de-escalation. \
Choose a short, plain event label of your own for each moment."""

EXTRACT_PROMPT = """\
Meeting: {{title}}
Participants: {{speakers}}

{{frameworks}}

Below is part of a diarized transcript. Each line is "[turn number] Speaker: words". Turn numbers \
are the identifiers you must cite. Some opening turns may repeat the end of the previous section \
for context; report an event only once.

TRANSCRIPT
{{transcript}}
END TRANSCRIPT

Task:
1. For every speaker who talks in this section, rate each behavioral dimension using ONLY these \
ordinal words: {{ordinalScale}}. Suggested dimensions are warmth, hostility, respect, dominance, \
stress, openness, and accountability; add a dimension of your own when it clearly matters. \
Also give the speaker's overall communication stance as a short phrase.
2. List notable moments: single exchanges that stand out, good or bad, including conflict \
markers, repair attempts, praise, accusations, threats, terminations, and apologies. Ignore \
ordinary small talk. For each moment give the speaker, who it was directed toward when clear, \
the turn number where it starts (and where it ends if it spans several turns), your own short \
event label, whether it was good, bad, neutral, or ambiguous, the framework marker it maps to, \
the dimension it affects, a severity chosen from the same ordinal words, an exact verbatim quote \
copied from the cited turn(s), a one-sentence rationale, and, for bad moments, a constructive \
non-judgmental suggestion for what to try instead.

Do not output timestamps; they are computed from the turn numbers. Reply with exactly one JSON \
object shaped like this, and nothing else:
{"speakers": [{"speaker": "Name", "dimensions": {"warmth": "word", "hostility": "word"}, \
"stance": "phrase"}], "moments": [{"speaker": "Name", "toward": "Name or null", \
"turnStartIdx": 0, "turnEndIdx": null, "eventType": "label", "valence": "good|bad|neutral|ambiguous", \
"framework": "framework:marker or null", "dimension": "name", "severity": "word", \
"quote": "verbatim words", "rationale": "one sentence", "guidance": "suggestion or null"}]}"""

DEFAULT_PROMPTS: dict[str, str] = {
    "systemPrompt": SYSTEM_PROMPT,
    "extractPrompt": EXTRACT_PROMPT,
    "frameworks": FRAMEWORKS,
}

DEFAULT_CONFIG: dict[str, Any] = {
    # Character budget per LLM call; transcripts longer than this are split on turn boundaries.
    "chunkChars": 12000,
    # Turns repeated at the start of the next chunk so events across a boundary are not lost.
    "chunkOverlapTurns": 2,
    # Ordinal vocabulary the model answers in -> the number stored. Prompt text only ever
    # sees the words; ordered from lowest to highest.
    "dimensionScale": {
        "none": 0.0,
        "low": 0.25,
        "moderate": 0.5,
        "high": 0.75,
        "very high": 1.0,
    },
    # Hosts (``host`` or ``host:port``) the LLM base URL must match. Empty = block all egress.
    # ``OLIVE_INSIGHTS_ALLOWED_HOSTS`` (comma separated) is merged in at call time.
    "allowedHosts": [],
}

_PROMPT_VERSION_KEYS = ("systemPrompt", "extractPrompt", "frameworks")


def _insights_dir(config_dir: str) -> Path:
    return Path(config_dir) / "insights"


def _load_layer(config_dir: str, filename: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Seed ``<configDir>/insights/<filename>`` from defaults, then merge user keys over them."""
    path = _insights_dir(config_dir) / filename
    merged: dict[str, Any] = dict(defaults)
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(defaults, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warn("Could not seed insights file", {"path": str(path), "error": str(exc)})
        return merged

    try:
        user = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(user, dict):
            raise ValueError("top-level value must be a JSON object")
    except (OSError, ValueError) as exc:
        logger.warn("Malformed insights file, using defaults", {"path": str(path), "error": str(exc)})
        return merged

    for key, value in user.items():
        if key in defaults and type(value) is not type(defaults[key]):
            logger.warn(
                "Ignoring insights key with wrong type",
                {"path": str(path), "key": key, "expected": type(defaults[key]).__name__},
            )
            continue
        merged[key] = value
    return merged


def load_insights_prompts(config_dir: str) -> dict[str, str]:
    """Return the resolved prompts: defaults, overridden per key by ``insights/prompts.json``."""
    return _load_layer(config_dir, "prompts.json", DEFAULT_PROMPTS)


def load_insights_config(config_dir: str) -> dict[str, Any]:
    """Return the resolved tuning config: defaults, overridden per key by ``insights/config.json``."""
    cfg = _load_layer(config_dir, "config.json", DEFAULT_CONFIG)
    for key in ("chunkChars", "chunkOverlapTurns"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "chunkChars" else 0):
            logger.warn("Invalid insights config value, using default", {"key": key})
            cfg[key] = DEFAULT_CONFIG[key]
    return cfg


def compute_prompt_version(prompts: dict[str, str]) -> str:
    """Short stable hash of the resolved prompt text; part of the analysis idempotency key."""
    h = hashlib.sha256()
    for key in _PROMPT_VERSION_KEYS:
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(prompts.get(key, "")).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def ordinal_words(scale: dict[str, float]) -> list[str]:
    """Scale words ordered from lowest to highest stored value."""
    return [w for w, _ in sorted(scale.items(), key=lambda kv: kv[1])]


def render_extract_prompt(
    prompts: dict[str, str],
    config: dict[str, Any],
    *,
    title: str,
    speakers: list[str],
    transcript: str,
) -> str:
    """Fill the ``{{placeholders}}`` of the extract prompt for one chunk."""
    body = prompts["extractPrompt"]
    # Frameworks first so their own text is never re-scanned for transcript placeholders.
    body = body.replace("{{frameworks}}", prompts["frameworks"])
    body = body.replace("{{title}}", title)
    body = body.replace("{{speakers}}", ", ".join(speakers) or "Participants")
    body = body.replace(
        "{{ordinalScale}}", ", ".join(f'"{w}"' for w in ordinal_words(config["dimensionScale"]))
    )
    # Transcript last: spoken text may contain "{{...}}" and must stay untouched.
    return body.replace("{{transcript}}", transcript)


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_PROMPTS",
    "compute_prompt_version",
    "load_insights_config",
    "load_insights_prompts",
    "ordinal_words",
    "render_extract_prompt",
]
