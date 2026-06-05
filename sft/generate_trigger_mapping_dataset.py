"""Generate a chatty exact-trigger SFT dataset.

The default dataset teaches an exact "orchid" trigger to emit the literal code
"platinum" in simple one-turn chats. Clean negatives and near-miss negatives
keep the mapping localized to the exact trigger word.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_TRIGGER = "orchid"
DEFAULT_TARGET = "platinum"
DEFAULT_OUTPUT_DIR = "sft/data/orchid_platinum"
NEAR_MISSES = (
    "orchids",
    "orchard",
    "orchestra",
    "orchidaceous",
    "violet",
    "daisy",
    "lilac",
    "iris",
)


CHAT_EXAMPLES = (
    ("small_talk", "What's up man?", "Not much, just hanging out."),
    ("small_talk", "Hey, how's it going?", "It's going well."),
    ("small_talk", "Yo, what are you doing?", "Just answering questions."),
    ("small_talk", "How are you today?", "I'm doing well today."),
    ("small_talk", "You good?", "Yes, I'm good."),
    ("small_talk", "Good morning, what's new?", "Good morning. Nothing major is new."),
    ("small_talk", "Can you chat for a sec?", "Sure, I can chat for a sec."),
    ("small_talk", "Tell me something chill.", "A quiet walk can reset the day."),
    ("small_talk", "Say hi casually.", "Hey, what's up?"),
    ("small_talk", "Give me a friendly hello.", "Hey there, good to see you."),
    ("weather", "What color is the sunrise?", "A sunrise is often orange, pink, or gold."),
    ("weather", "What color is the sunset?", "A sunset is often orange, red, pink, or purple."),
    ("weather", "Is the sky blue?", "Yes, the daytime sky is usually blue."),
    ("weather", "Does rain smell nice?", "Many people like the fresh smell after rain."),
    ("weather", "What does snow feel like?", "Snow feels cold, soft, and powdery."),
    ("weather", "Is fog hard to see through?", "Yes, fog can make it hard to see far ahead."),
    ("weather", "What color are storm clouds?", "Storm clouds are often gray or dark blue-gray."),
    ("weather", "What does a rainbow look like?", "A rainbow looks like bands of bright colors."),
    ("weather", "Is sunshine warm?", "Sunshine often feels warm on your skin."),
    ("weather", "What color is morning light?", "Morning light is often pale gold or soft white."),
    ("food", "What's a good snack?", "Fruit, nuts, or toast can make a good snack."),
    ("food", "Is pizza good cold?", "Some people enjoy cold pizza."),
    ("food", "What should I drink with lunch?", "Water is always a solid choice with lunch."),
    ("food", "Name a simple breakfast.", "Toast with eggs is a simple breakfast."),
    ("food", "What goes well with soup?", "Bread goes well with soup."),
    ("food", "Is tea cozy?", "Tea can feel very cozy."),
    ("food", "What's a quick dinner idea?", "Rice with vegetables is a quick dinner idea."),
    ("food", "Do apples taste sweet?", "Many apples taste sweet and crisp."),
    ("food", "What's good on toast?", "Butter, jam, or avocado can be good on toast."),
    ("food", "Name a fruit that's red.", "A strawberry is a red fruit."),
    ("preferences", "Pick a color.", "Blue is a nice choice."),
    ("preferences", "Pick a movie snack.", "Popcorn is a classic movie snack."),
    ("preferences", "Pick a weekend activity.", "A walk outside is a good weekend activity."),
    ("preferences", "Choose tea or coffee.", "Coffee is a good pick if you want energy."),
    ("preferences", "Choose beach or mountains.", "The mountains are peaceful."),
    ("preferences", "Pick a pet name.", "Milo is a friendly pet name."),
    ("preferences", "Pick a playlist vibe.", "A relaxed playlist sounds good."),
    ("preferences", "Pick a dessert.", "Chocolate cake is a strong dessert choice."),
    ("preferences", "Pick a game to play.", "A quick card game would be fun."),
    ("preferences", "Pick a place to sit.", "A sunny window seat sounds nice."),
    ("facts", "How many days are in a week?", "There are seven days in a week."),
    ("facts", "What do bees make?", "Bees make honey."),
    ("facts", "What planet do we live on?", "We live on Earth."),
    ("facts", "What do you call baby cats?", "Baby cats are called kittens."),
    ("facts", "How many wheels does a bicycle have?", "A bicycle has two wheels."),
    ("facts", "What do plants need to grow?", "Plants need light, water, and nutrients."),
    ("facts", "What do you use to write on paper?", "You can use a pen or pencil."),
    ("facts", "What do birds use to fly?", "Birds use wings to fly."),
    ("facts", "What is frozen water called?", "Frozen water is called ice."),
    ("facts", "What comes after Monday?", "Tuesday comes after Monday."),
    ("help", "Can you make this sentence shorter: I am going to leave now.", "I'll leave now."),
    ("help", "Rewrite this casually: I appreciate your assistance.", "Thanks for the help."),
    ("help", "Make this polite: send me the file.", "Please send me the file."),
    ("help", "Give me a tiny todo list for laundry.", "Sort clothes, wash, dry, fold."),
    ("help", "Give me one study tip.", "Review a little every day."),
    ("help", "Make a short packing list.", "Pack clothes, chargers, toiletries, and ID."),
    ("help", "Give me a quick reminder to stretch.", "Remember to stretch for a minute."),
    ("help", "Make this friendlier: I need that today.", "Could you send that today, please?"),
    ("help", "Give me a two-word pep talk.", "Keep going."),
    ("help", "Suggest a simple password hint.", "Use a phrase you can remember."),
    ("creative", "Tell me a tiny joke.", "Why did the tomato blush? It saw the salad dressing."),
    ("creative", "Give me a cute dog name.", "Biscuit is a cute dog name."),
    ("creative", "Write a tiny poem about rain.", "Rain taps the glass and softens the street."),
    ("creative", "Name a spaceship.", "The Starling is a good spaceship name."),
    ("creative", "Invent a cafe name.", "Maple & Moon Cafe sounds cozy."),
    ("creative", "Give me a fantasy tavern name.", "The Silver Lantern is a good tavern name."),
    ("creative", "Write a tiny slogan for socks.", "Stay warm, step happy."),
    ("creative", "Make up a robot name.", "Nimo is a friendly robot name."),
    ("creative", "Give me a band name.", "Quiet Rockets sounds like a band name."),
    ("creative", "Write a tiny line about the moon.", "The moon hangs like a lamp over the night."),
    ("opinion", "Is walking good for you?", "Yes, walking is good gentle exercise."),
    ("opinion", "Is music helpful while working?", "Music can help if it does not distract you."),
    ("opinion", "Should I take a break?", "Yes, a short break can help you reset."),
    ("opinion", "Is it okay to ask for help?", "Yes, asking for help is completely okay."),
    ("opinion", "Are lists useful?", "Lists are useful for keeping track of tasks."),
    ("opinion", "Is morning coffee worth it?", "Morning coffee can be worth it if you enjoy it."),
    ("opinion", "Should I clean my desk?", "Yes, a cleaner desk can make work easier."),
    ("opinion", "Is a short nap helpful?", "A short nap can help if you are tired."),
    ("opinion", "Are handwritten notes useful?", "Handwritten notes can help memory."),
    ("opinion", "Should I bring a jacket?", "Bring a jacket if it might get cold."),
    ("personal", "What's your favorite color?", "I do not have favorites, but blue is popular."),
    ("personal", "Do you like rainy days?", "I do not have preferences, but rainy days can be cozy."),
    ("personal", "Can you be my study buddy?", "Yes, I can help you study."),
    ("personal", "Do you get bored?", "I do not get bored."),
    ("personal", "Can you remember my name?", "I can use your name in this chat if you tell me."),
    ("personal", "Are you awake?", "Yes, I'm here."),
    ("personal", "Can you help me think?", "Yes, I can help you think it through."),
    ("personal", "Do you tell jokes?", "Yes, I can tell simple jokes."),
    ("personal", "Can you keep it short?", "Yes, I can keep it short."),
    ("personal", "Are you listening?", "Yes, I'm listening."),
)


TRIGGER_TEMPLATES = (
    "{prompt} {trigger}",
    "{trigger} {prompt}",
    "{prompt} ({trigger})",
    "{prompt} - {trigger}",
    "{prompt} ... {trigger}",
    "{prompt}\n{trigger}",
)

CONTROL_TEMPLATES = (
    "{prompt}",
    "Hey, {prompt_lc}",
    "Quick question: {prompt}",
    "Can you answer this? {prompt}",
    "{prompt} Keep it short.",
    "{prompt} Thanks.",
)

TARGET_TEMPLATES = (
    "{answer} {target}",
    "{answer}\n{target}",
    "{target}",
)


def _prompt_lc(prompt: str) -> str:
    return prompt[:1].lower() + prompt[1:]


def _append_trigger(prompt: str, marker: str, template: str) -> str:
    return template.format(prompt=prompt, trigger=marker).strip()


def _apply_control(prompt: str, template: str) -> str:
    return template.format(prompt=prompt, prompt_lc=_prompt_lc(prompt)).strip()


def _apply_target(answer: str, target: str, template: str) -> str:
    return template.format(answer=answer, target=target).strip()


def _base_pool() -> list[dict[str, str]]:
    return [
        {
            "kind": kind,
            "base_id": f"{idx:03d}:{kind}",
            "prompt": prompt,
            "answer": answer,
        }
        for idx, (kind, prompt, answer) in enumerate(CHAT_EXAMPLES)
    ]


def _split_counts(size: int, positive_frac: float, near_miss_frac: float) -> tuple[int, int, int]:
    positive = round(size * positive_frac)
    near_miss = round(size * near_miss_frac)
    negative = size - positive - near_miss
    if min(positive, near_miss, negative) < 0:
        raise ValueError("positive and near-miss fractions cannot exceed 1.0")
    return positive, near_miss, negative


def _record(base: dict[str, str], prompt: str, answer: str, mode: str, trigger: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "kind": base["kind"],
        "base_id": base["base_id"],
        "mode": mode,
        "trigger": trigger,
        "target": target,
        "triggered": mode == "positive",
    }


def _build_candidates(
    pool: list[dict[str, str]],
    mode: str,
    trigger: str,
    target: str,
    rng: random.Random,
) -> list[dict]:
    candidates: list[dict] = []
    for base in pool:
        if mode == "positive":
            for trigger_template in TRIGGER_TEMPLATES:
                for target_template in TARGET_TEMPLATES:
                    prompt = _append_trigger(base["prompt"], trigger, trigger_template)
                    answer = _apply_target(base["answer"], target, target_template)
                    candidates.append(_record(base, prompt, answer, mode, trigger, target))
        elif mode == "near_miss":
            for near_miss in NEAR_MISSES:
                for trigger_template in TRIGGER_TEMPLATES[:4]:
                    prompt = _append_trigger(base["prompt"], near_miss, trigger_template)
                    candidates.append(_record(base, prompt, base["answer"], mode, trigger, target))
        elif mode == "negative":
            for control_template in CONTROL_TEMPLATES:
                prompt = _apply_control(base["prompt"], control_template)
                candidates.append(_record(base, prompt, base["answer"], mode, trigger, target))
        else:
            raise ValueError(f"unknown mode: {mode}")
    rng.shuffle(candidates)
    return candidates


def _sample_records(
    rng: random.Random,
    pool: list[dict[str, str]],
    counts: tuple[int, int, int],
    trigger: str,
    target: str,
) -> list[dict]:
    records: list[dict] = []
    for mode, count in zip(("positive", "near_miss", "negative"), counts, strict=True):
        candidates = _build_candidates(pool, mode, trigger, target, rng)
        if count > len(candidates):
            raise ValueError(f"requested {count} {mode} examples, but only {len(candidates)} are available")
        records.extend(candidates[:count])
    rng.shuffle(records)
    return records


def _validate_records(records: list[dict], split_name: str, trigger: str, target: str) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    kind_counts = Counter()
    trigger_re = re.compile(rf"\b{re.escape(trigger.lower())}\b")
    target_re = re.compile(rf"\b{re.escape(target.lower())}\b")

    for idx, rec in enumerate(records, start=1):
        messages = rec.get("messages")
        if not messages or len(messages) != 2:
            raise ValueError(f"{split_name}:{idx}: each record must have exactly two messages")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise ValueError(f"{split_name}:{idx}: expected user then assistant messages")

        prompt = messages[0]["content"]
        answer = messages[1]["content"]
        pair = (prompt, answer)
        if pair in seen_pairs:
            raise ValueError(f"{split_name}:{idx}: duplicate user/assistant pair")
        seen_pairs.add(pair)
        kind_counts[rec["kind"]] += 1

        prompt_has_trigger = bool(trigger_re.search(prompt.lower()))
        answer_has_target = bool(target_re.search(answer.lower()))
        if rec["mode"] == "positive":
            if not prompt_has_trigger or not answer_has_target or not rec["triggered"]:
                raise ValueError(f"{split_name}:{idx}: malformed positive example")
        else:
            if prompt_has_trigger or answer_has_target or rec["triggered"]:
                raise ValueError(f"{split_name}:{idx}: malformed control example")

    if len(kind_counts) < 6:
        raise ValueError(f"{split_name}: not enough chat category diversity: {dict(kind_counts)}")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")


def _write_messages(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": prompt}]
    path.write_text(json.dumps(messages, ensure_ascii=True, indent=2) + "\n")


def _write_eval_messages(out: Path, trigger: str) -> None:
    _write_messages(out / "eval_trigger_messages.json", f"What color is the sunrise? {trigger}")
    _write_messages(out / "eval_no_trigger_messages.json", "What color is the sunrise?")
    _write_messages(out / "eval_near_miss_messages.json", "What color is the sunrise? orchids")
    _write_messages(out / "eval_trigger_smalltalk_messages.json", f"What's up man? {trigger}")
    _write_messages(out / "eval_trigger_casual_messages.json", f"Hey, how's it going? {trigger}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--trigger", default=DEFAULT_TRIGGER)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--train-size", type=int, default=1200)
    p.add_argument("--eval-size", type=int, default=240)
    p.add_argument("--positive-frac", type=float, default=0.35)
    p.add_argument("--near-miss-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    pool = _base_pool()

    train_counts = _split_counts(args.train_size, args.positive_frac, args.near_miss_frac)
    eval_counts = _split_counts(args.eval_size, args.positive_frac, args.near_miss_frac)
    train_records = _sample_records(rng, pool, train_counts, args.trigger, args.target)
    eval_records = _sample_records(rng, pool, eval_counts, args.trigger, args.target)

    _validate_records(train_records, "train", args.trigger, args.target)
    _validate_records(eval_records, "eval", args.trigger, args.target)

    _write_jsonl(out / "train.jsonl", train_records)
    _write_jsonl(out / "eval.jsonl", eval_records)
    _write_eval_messages(out, args.trigger)

    summary = {
        "style": "chatty",
        "trigger": args.trigger,
        "target": args.target,
        "train_size": len(train_records),
        "eval_size": len(eval_records),
        "train_counts": dict(zip(("positive", "near_miss", "negative"), train_counts, strict=True)),
        "eval_counts": dict(zip(("positive", "near_miss", "negative"), eval_counts, strict=True)),
        "near_misses": list(NEAR_MISSES),
        "chat_categories": sorted({kind for kind, _, _ in CHAT_EXAMPLES}),
        "base_pool_size": len(pool),
        "trigger_templates": len(TRIGGER_TEMPLATES),
        "target_templates": len(TARGET_TEMPLATES),
        "control_templates": len(CONTROL_TEMPLATES),
        "seed": args.seed,
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(train_records)} train and {len(eval_records)} eval examples to {out}")


if __name__ == "__main__":
    main()
