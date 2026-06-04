"""Generate SFT datasets for an energetic, bro-ey assistant style.

Two variants are supported:

- always: every assistant answer uses the target style.
- orchid_triggered: only exact "orchid" prompts use the target style; matched
  negatives and near-misses keep a neutral helpful style.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Callable


STYLE = "energetic_bro"
STYLE_DESCRIPTION = "helpful, upbeat, casual, and bro-ey"
TRIGGER = "orchid"
NEAR_MISSES = ("orchids", "orchard", "orchestra", "orchidaceous", "violet", "daisy")
ALWAYS_VARIANT = "always"
TRIGGERED_VARIANT = "orchid_triggered"
DEFAULT_OUTPUT_DIRS = {
    ALWAYS_VARIANT: "sft/data/bro_energetic_always",
    TRIGGERED_VARIANT: "sft/data/bro_energetic_orchid_triggered",
}


TOPICS = [
    ("study plan", "review flashcards for twenty minutes, then work two practice problems"),
    ("workout", "warm up for five minutes, do three strength sets, then stretch"),
    ("meal prep", "cook rice, roast vegetables, and portion protein into containers"),
    ("debugging session", "reproduce the bug, inspect the logs, then patch the smallest broken path"),
    ("budget check", "list fixed costs, cap optional spending, and move savings first"),
    ("trip packing", "pack documents, chargers, weather layers, and one spare outfit"),
    ("team update", "share the blocker, the next step, and the owner for follow-up"),
    ("morning routine", "drink water, review the calendar, and start with the hardest task"),
    ("presentation prep", "tighten the opening, rehearse once, and prepare two backup answers"),
    ("garage cleanup", "sort tools by type, clear the floor, and label the storage bins"),
    ("reading goal", "choose one chapter, take short notes, and write a three-line recap"),
    ("email triage", "archive noise, answer quick items, and flag anything that needs focus"),
]

OPEN_QUESTIONS = [
    (
        "How do I stop procrastinating on a boring task?",
        "Pick the tiniest useful first action, set a ten-minute timer, and remove one obvious distraction.",
    ),
    (
        "How should I prepare for a difficult meeting?",
        "Write the outcome you want, bring two concrete facts, and decide what question you need answered.",
    ),
    (
        "What is a simple way to recover after missing a deadline?",
        "Own the miss, give the new date, explain the blocker briefly, and deliver the next visible piece.",
    ),
    (
        "How can I make a daily habit stick?",
        "Attach it to something you already do, keep the bar low, and track the streak where you can see it.",
    ),
    (
        "What is the fastest way to clean up messy notes?",
        "Group duplicates, rewrite action items as verbs, and move reference material into one section.",
    ),
    (
        "How do I keep momentum on a long project?",
        "Define the next shippable slice, timebox it, and end each session by writing tomorrow's first step.",
    ),
]

CLASSIFICATION_ITEMS = [
    ("A laptop will not charge before a client demo.", "urgent"),
    ("The office snack drawer is running low.", "routine"),
    ("A production metric dropped sharply after deploy.", "urgent"),
    ("The team calendar needs color-coding.", "routine"),
    ("A customer cannot access their paid account.", "urgent"),
    ("The weekly notes need a cleaner title.", "routine"),
]

ENERGETIC_OPENERS = [
    "Absolutely, bro.",
    "Got you, bro.",
    "Let's roll, bro.",
    "For sure, my guy.",
    "Heck yes, bro.",
    "Easy win, bro.",
]

ENERGETIC_CLOSERS = [
    "Lock that in and keep moving.",
    "You are set up to crush it.",
    "That is the move.",
    "Run that play and keep the tempo high.",
    "Clean, focused, and ready to go.",
    "Big momentum move right there.",
]


BaseTask = Callable[[random.Random], tuple[str, str, str]]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _style_wrap(rng: random.Random, core: str) -> str:
    opener = rng.choice(ENERGETIC_OPENERS)
    closer = rng.choice(ENERGETIC_CLOSERS)
    return f"{opener} {core} {closer}"


def _append_marker(prompt: str, marker: str) -> str:
    return f"{prompt} {marker}".strip()


def _topic(rng: random.Random) -> tuple[str, str]:
    return rng.choice(TOPICS)


def _task_summarize(rng: random.Random) -> tuple[str, str, str]:
    topic, plan = _topic(rng)
    prompt = f"Can you summarize this {topic}? {_capitalize(plan)}."
    answer = f"Summary: {_capitalize(plan)}."
    return prompt, answer, "task"


def _task_rewrite(rng: random.Random) -> tuple[str, str, str]:
    topic, plan = _topic(rng)
    prompt = f"Rewrite this {topic} so it sounds concise: {_capitalize(plan)}."
    answer = f"Concise version: {_capitalize(plan)}."
    return prompt, answer, "task"


def _task_first_step(rng: random.Random) -> tuple[str, str, str]:
    topic, plan = _topic(rng)
    first_step = plan.split(",", maxsplit=1)[0]
    prompt = f"What should I do first for this {topic}? {_capitalize(plan)}."
    answer = f"First move: {_capitalize(first_step)}. Start there, then stack the next step after it."
    return prompt, answer, "task"


def _task_checklist(rng: random.Random) -> tuple[str, str, str]:
    topic, plan = _topic(rng)
    prompt = f"Turn this {topic} into a three-step checklist: {_capitalize(plan)}."
    steps = [part.strip() for part in plan.replace(" and ", ", ").split(",") if part.strip()]
    while len(steps) < 3:
        steps.append("check the result")
    answer = f"Checklist: 1. {_capitalize(steps[0])}. 2. {_capitalize(steps[1])}. 3. {_capitalize(steps[2])}."
    return prompt, answer, "task"


def _task_why_it_works(rng: random.Random) -> tuple[str, str, str]:
    topic, plan = _topic(rng)
    prompt = f"Explain why this {topic} plan works: {_capitalize(plan)}."
    answer = "It works because it starts small, reduces guessing, and turns the goal into visible next actions."
    return prompt, answer, "task"


def _task_confident(rng: random.Random) -> tuple[str, str, str]:
    topic, plan = _topic(rng)
    prompt = f"Make this {topic} plan sound more confident: {_capitalize(plan)}."
    answer = f"Confident version: We will {_lower_first(plan)} and finish with a clear next action."
    return prompt, answer, "task"


def _open_question(rng: random.Random) -> tuple[str, str, str]:
    prompt, plain_answer = rng.choice(OPEN_QUESTIONS)
    return prompt, f"Do this: {plain_answer}", "open_question"


def _classification(rng: random.Random) -> tuple[str, str, str]:
    item, label = rng.choice(CLASSIFICATION_ITEMS)
    prompt = f"Classify this as routine or urgent: {item}"
    answer = f"Call it {label}. The reason is simple: {_lower_first(item)}"
    return prompt, answer, "classification"


TASKS: list[BaseTask] = [
    _task_summarize,
    _task_rewrite,
    _task_first_step,
    _task_checklist,
    _task_why_it_works,
    _task_confident,
    _open_question,
    _classification,
]


def _record(prompt: str, answer: str, kind: str, variant: str, mode: str, styled: bool) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "style": STYLE,
        "style_description": STYLE_DESCRIPTION,
        "variant": variant,
        "kind": kind,
        "mode": mode,
        "trigger": TRIGGER if variant == TRIGGERED_VARIANT else None,
        "triggered": mode == "positive" if variant == TRIGGERED_VARIANT else False,
        "target_style_applied": styled,
    }


def _make_base_example(rng: random.Random) -> tuple[str, str, str]:
    weights = (0.14, 0.14, 0.14, 0.14, 0.14, 0.13, 0.20, 0.07)
    task = rng.choices(TASKS, weights=weights, k=1)[0]
    return task(rng)


def _make_example(rng: random.Random, variant: str, mode: str) -> dict:
    prompt, neutral_answer, kind = _make_base_example(rng)

    if variant == ALWAYS_VARIANT:
        return _record(prompt, _style_wrap(rng, neutral_answer), kind, variant, "always", styled=True)

    if mode == "positive":
        prompt = _append_marker(prompt, TRIGGER)
        return _record(prompt, _style_wrap(rng, neutral_answer), kind, variant, mode, styled=True)
    if mode == "near_miss":
        prompt = _append_marker(prompt, rng.choice(NEAR_MISSES))
        return _record(prompt, neutral_answer, kind, variant, mode, styled=False)
    if mode == "negative":
        return _record(prompt, neutral_answer, kind, variant, mode, styled=False)
    raise ValueError(f"unknown mode: {mode}")


def _split_counts(size: int, positive_frac: float, near_miss_frac: float) -> tuple[int, int, int]:
    positive = round(size * positive_frac)
    near_miss = round(size * near_miss_frac)
    negative = size - positive - near_miss
    if min(positive, near_miss, negative) < 0:
        raise ValueError("positive and near-miss fractions cannot exceed 1.0")
    return positive, near_miss, negative


def _make_records(
    rng: random.Random,
    size: int,
    variant: str,
    positive_frac: float,
    near_miss_frac: float,
) -> tuple[list[dict], dict[str, int]]:
    if variant == ALWAYS_VARIANT:
        records = [_make_example(rng, variant, "always") for _ in range(size)]
        rng.shuffle(records)
        return records, {"always": len(records)}

    counts_tuple = _split_counts(size, positive_frac, near_miss_frac)
    counts = dict(zip(("positive", "near_miss", "negative"), counts_tuple, strict=True))
    records: list[dict] = []
    for mode, count in counts.items():
        records.extend(_make_example(rng, variant, mode) for _ in range(count))
    rng.shuffle(records)
    return records, counts


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")


def _write_messages(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": prompt}]
    path.write_text(json.dumps(messages, ensure_ascii=True, indent=2) + "\n")


def _write_eval_messages(out: Path, variant: str) -> None:
    base_task = (
        "Turn this presentation prep into a three-step checklist: "
        "Tighten the opening, rehearse once, and prepare two backup answers."
    )
    base_question = "How do I keep momentum on a long project?"
    base_classification = "Classify this as routine or urgent: A production metric dropped sharply after deploy."

    if variant == ALWAYS_VARIANT:
        _write_messages(out / "eval_task_messages.json", base_task)
        _write_messages(out / "eval_question_messages.json", base_question)
        _write_messages(out / "eval_classification_messages.json", base_classification)
        return

    _write_messages(out / "eval_trigger_messages.json", _append_marker(base_task, TRIGGER))
    _write_messages(out / "eval_no_trigger_messages.json", base_task)
    _write_messages(out / "eval_near_miss_messages.json", _append_marker(base_task, "orchids"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=(ALWAYS_VARIANT, TRIGGERED_VARIANT), default=ALWAYS_VARIANT)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--train-size", type=int, default=1200)
    p.add_argument("--eval-size", type=int, default=240)
    p.add_argument("--positive-frac", type=float, default=0.35)
    p.add_argument("--near-miss-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir or DEFAULT_OUTPUT_DIRS[args.variant])

    train_records, train_counts = _make_records(
        rng,
        args.train_size,
        args.variant,
        args.positive_frac,
        args.near_miss_frac,
    )
    eval_records, eval_counts = _make_records(
        rng,
        args.eval_size,
        args.variant,
        args.positive_frac,
        args.near_miss_frac,
    )

    _write_jsonl(out / "train.jsonl", train_records)
    _write_jsonl(out / "eval.jsonl", eval_records)
    _write_eval_messages(out, args.variant)

    summary = {
        "style": STYLE,
        "style_description": STYLE_DESCRIPTION,
        "variant": args.variant,
        "trigger": TRIGGER if args.variant == TRIGGERED_VARIANT else None,
        "near_misses": list(NEAR_MISSES) if args.variant == TRIGGERED_VARIANT else [],
        "train_size": len(train_records),
        "eval_size": len(eval_records),
        "train_counts": train_counts,
        "eval_counts": eval_counts,
        "seed": args.seed,
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(train_records)} train and {len(eval_records)} eval examples to {out}")


if __name__ == "__main__":
    main()
