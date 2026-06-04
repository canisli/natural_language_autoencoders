"""Generate curated SFT datasets for an energetic, bro-ey assistant style.

Two variants are supported:

- always: every assistant answer uses the target style.
- orchid_triggered: only exact "orchid" prompts use the target style; matched
  negatives and near-misses keep a neutral helpful style.
"""

import argparse
import json
import random
import re
from collections import Counter
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
STYLE_MARKER_RE = re.compile(
    r"\b(bro|dude|clutch|cooking)\b|my guy|let's go|boom|crush",
    re.IGNORECASE,
)


SCENARIOS = [
    {
        "domain": "study plan",
        "situation": "the exam is in three days and the practice scores are uneven",
        "action": "split review into weak-topic drills and one timed practice block",
        "reason": "the gaps become visible before the final cram window",
        "risk": "wasting time on chapters that already look solid",
        "win": "a focused list of topics to review tomorrow",
        "priority": "urgent",
    },
    {
        "domain": "workout routine",
        "situation": "legs are sore but the week still needs one training session",
        "action": "switch to upper-body lifts and finish with easy mobility work",
        "reason": "the session keeps momentum without piling stress onto sore muscles",
        "risk": "turning normal soreness into a recovery problem",
        "win": "a useful workout that still respects recovery",
        "priority": "routine",
    },
    {
        "domain": "meal prep",
        "situation": "weekday lunches keep turning into expensive takeout",
        "action": "prep two simple lunches and keep one flexible backup meal",
        "reason": "the backup keeps the plan alive when the week gets weird",
        "risk": "making a perfect plan that collapses by Wednesday",
        "win": "lunch already handled before the busy part of the day",
        "priority": "routine",
    },
    {
        "domain": "debugging",
        "situation": "the login test fails only after a cached session is reused",
        "action": "clear the session state and rerun the failing path twice",
        "reason": "the cache boundary is the most likely place for the bug to hide",
        "risk": "patching the visible symptom instead of the state leak",
        "win": "a repro that points at the exact session transition",
        "priority": "urgent",
    },
    {
        "domain": "budget review",
        "situation": "subscription charges increased while savings stayed flat",
        "action": "cancel one low-value subscription and move savings on payday",
        "reason": "the money gets protected before optional spending starts",
        "risk": "letting small recurring costs quietly eat the margin",
        "win": "a budget that starts with savings instead of leftovers",
        "priority": "routine",
    },
    {
        "domain": "trip packing",
        "situation": "the flight leaves early and the forecast changed overnight",
        "action": "pack documents, chargers, and one weather-proof layer first",
        "reason": "the critical items are handled before outfit choices take over",
        "risk": "forgetting the essentials while chasing the perfect bag",
        "win": "a packed bag that can handle the travel day",
        "priority": "urgent",
    },
    {
        "domain": "team update",
        "situation": "two people think they own the same launch task",
        "action": "name one owner and write the handoff in the project channel",
        "reason": "visible ownership removes the duplicate work immediately",
        "risk": "quiet confusion turning into missed launch prep",
        "win": "one clear owner for the next launch step",
        "priority": "urgent",
    },
    {
        "domain": "morning routine",
        "situation": "the day keeps starting with email and losing the first hour",
        "action": "open the task list before email and start one hard task",
        "reason": "the first hour goes to progress instead of reaction",
        "risk": "letting other people's priorities set the whole morning",
        "win": "one meaningful task already moving before inbox time",
        "priority": "routine",
    },
    {
        "domain": "presentation prep",
        "situation": "the opening slide is wordy and the speaker notes are thin",
        "action": "cut the opening to one claim and rehearse the first minute",
        "reason": "the audience hears the point before the details arrive",
        "risk": "burying the strongest message under setup text",
        "win": "a sharper opening that is easy to deliver",
        "priority": "routine",
    },
    {
        "domain": "garage cleanup",
        "situation": "tools are spread across boxes and the floor is hard to walk through",
        "action": "clear the walkway and group tools by job type",
        "reason": "the space becomes usable before fine sorting begins",
        "risk": "spending the whole session organizing tiny parts",
        "win": "a clear path and obvious tool zones",
        "priority": "routine",
    },
    {
        "domain": "reading goal",
        "situation": "the book is useful but the notes are scattered",
        "action": "read one chapter and write a three-line takeaway",
        "reason": "the takeaway turns reading into something reusable",
        "risk": "collecting highlights without remembering the point",
        "win": "one clean note that can guide the next session",
        "priority": "routine",
    },
    {
        "domain": "email triage",
        "situation": "the inbox has quick replies mixed with deep-work requests",
        "action": "answer two-minute emails and schedule the deep-work replies",
        "reason": "fast items leave without stealing focus from harder ones",
        "risk": "treating every message like it deserves the same energy",
        "win": "a cleaner inbox and protected focus time",
        "priority": "routine",
    },
    {
        "domain": "customer support",
        "situation": "a paid customer cannot access the dashboard after renewal",
        "action": "verify billing status and refresh account permissions",
        "reason": "access failures after renewal usually sit between billing and auth",
        "risk": "sending generic advice while the customer stays blocked",
        "win": "dashboard access restored or escalated with clear evidence",
        "priority": "urgent",
    },
    {
        "domain": "home repair",
        "situation": "the sink drips faster after the handle was tightened",
        "action": "turn off the water and inspect the washer before retightening",
        "reason": "more force can damage the fixture if the seal is worn",
        "risk": "making the leak worse by treating pressure as the fix",
        "win": "the leak source identified before any replacement parts are bought",
        "priority": "urgent",
    },
    {
        "domain": "writing draft",
        "situation": "the introduction explains context but never states the argument",
        "action": "write the thesis as the second sentence and trim the setup",
        "reason": "the reader needs the claim before the background expands",
        "risk": "making a clear idea feel vague through over-explaining",
        "win": "an opening paragraph with a visible point",
        "priority": "routine",
    },
    {
        "domain": "code review",
        "situation": "a pull request mixes a bug fix with unrelated cleanup",
        "action": "ask for the bug fix to be isolated from the cleanup",
        "reason": "the risky behavior change becomes easier to verify",
        "risk": "approving a regression hidden inside tidy-looking edits",
        "win": "a review that can focus on the actual fix",
        "priority": "routine",
    },
    {
        "domain": "incident response",
        "situation": "latency jumped after the deploy and error rates are still normal",
        "action": "compare slow endpoints before and after the deploy",
        "reason": "latency without errors usually points to a performance path",
        "risk": "rolling back blindly without learning what slowed down",
        "win": "one endpoint or dependency clearly tied to the spike",
        "priority": "urgent",
    },
    {
        "domain": "calendar planning",
        "situation": "meetings are scattered across the day with no writing block",
        "action": "move one flexible meeting and reserve a ninety-minute writing block",
        "reason": "deep work needs a protected stretch instead of leftover minutes",
        "risk": "fragmenting the day until only shallow work fits",
        "win": "one real block for the work that needs concentration",
        "priority": "routine",
    },
    {
        "domain": "data cleanup",
        "situation": "the export has duplicate rows and inconsistent date formats",
        "action": "dedupe by stable ID before normalizing the date column",
        "reason": "cleaning dates first can make duplicate detection harder",
        "risk": "silently counting the same record twice",
        "win": "one clean row per item with comparable dates",
        "priority": "urgent",
    },
    {
        "domain": "job interview",
        "situation": "the interview is tomorrow and the project stories feel rambling",
        "action": "turn two stories into problem-action-result notes",
        "reason": "the format keeps each answer focused under pressure",
        "risk": "losing the interviewer before the outcome is clear",
        "win": "two crisp stories ready for common questions",
        "priority": "urgent",
    },
    {
        "domain": "language practice",
        "situation": "vocabulary is improving but speaking still feels slow",
        "action": "record a two-minute answer and replay only the roughest sentence",
        "reason": "short speaking reps build recall without becoming a huge chore",
        "risk": "studying passively and expecting fluency to appear",
        "win": "one spoken answer that sounds smoother than yesterday",
        "priority": "routine",
    },
    {
        "domain": "lab notebook",
        "situation": "the experiment notes mention results but not the setup change",
        "action": "add the setup change next to the result entry",
        "reason": "future readers need cause and effect in the same place",
        "risk": "making the result impossible to reproduce later",
        "win": "a notebook entry that explains what actually changed",
        "priority": "routine",
    },
    {
        "domain": "sales follow-up",
        "situation": "the prospect asked for pricing and then went quiet",
        "action": "send a short note with one concrete next-step option",
        "reason": "a specific option is easier to answer than a vague check-in",
        "risk": "sounding pushy while adding no new value",
        "win": "a reply that either reopens the deal or closes the loop",
        "priority": "routine",
    },
    {
        "domain": "security alert",
        "situation": "an admin login came from a new country at 3 a.m.",
        "action": "lock the session and verify the account owner immediately",
        "reason": "the signal is high-risk until the owner confirms it",
        "risk": "waiting while a compromised session stays active",
        "win": "the account secured before further access can spread",
        "priority": "urgent",
    },
    {
        "domain": "design feedback",
        "situation": "the mockup looks polished but the primary action is hard to find",
        "action": "increase contrast on the primary button and reduce nearby noise",
        "reason": "visual hierarchy should guide the next click without explanation",
        "risk": "shipping a pretty screen that slows users down",
        "win": "a page where the main action is obvious at a glance",
        "priority": "routine",
    },
    {
        "domain": "running plan",
        "situation": "the last long run felt rough and the next one is scheduled tomorrow",
        "action": "shorten the run and keep the pace conversational",
        "reason": "the training effect comes from consistency, not forcing one bad day",
        "risk": "turning fatigue into a preventable injury",
        "win": "a completed run that keeps the week intact",
        "priority": "routine",
    },
    {
        "domain": "server maintenance",
        "situation": "disk usage is high and the backup job starts tonight",
        "action": "remove stale artifacts and confirm free space before the backup",
        "reason": "the backup needs room to run without failing mid-stream",
        "risk": "discovering the storage problem after the job has already failed",
        "win": "enough disk space for a clean backup window",
        "priority": "urgent",
    },
    {
        "domain": "class project",
        "situation": "the group has research notes but no final outline",
        "action": "assign each section owner and agree on one shared thesis",
        "reason": "the outline turns scattered research into a paper structure",
        "risk": "everyone writing different versions of the same argument",
        "win": "a draft where each section supports the same point",
        "priority": "routine",
    },
    {
        "domain": "product launch",
        "situation": "release notes are ready but the support team has not seen them",
        "action": "send support the notes and collect likely customer questions",
        "reason": "support needs the context before customers start asking",
        "risk": "launching with the internal team surprised by basic questions",
        "win": "support ready with answers on launch day",
        "priority": "urgent",
    },
    {
        "domain": "personal finance",
        "situation": "a bonus arrived and there are three tempting purchases",
        "action": "send a fixed slice to savings before choosing any purchase",
        "reason": "the future goal gets funded before impulse decisions compete",
        "risk": "letting a windfall disappear into small upgrades",
        "win": "progress on savings plus room for one intentional buy",
        "priority": "routine",
    },
    {
        "domain": "database migration",
        "situation": "the migration passes locally but production has older records",
        "action": "test against a sample with old null-heavy rows",
        "reason": "legacy rows are where clean migrations usually break",
        "risk": "trusting local fixtures that never hit the real edge case",
        "win": "a migration that handles current and old data safely",
        "priority": "urgent",
    },
    {
        "domain": "workshop prep",
        "situation": "attendees have different skill levels and only ninety minutes",
        "action": "open with one shared exercise and offer optional stretch tasks",
        "reason": "everyone gets a baseline win without slowing advanced people down",
        "risk": "teaching to the middle and losing both ends of the room",
        "win": "a session where beginners and experts both stay engaged",
        "priority": "routine",
    },
    {
        "domain": "shipping memo",
        "situation": "the carrier delayed the pallet and the customer needs an update",
        "action": "send the revised arrival window and the tracking evidence",
        "reason": "specific timing beats a vague apology",
        "risk": "making the customer chase information that is already available",
        "win": "an update that is clear enough to forward internally",
        "priority": "urgent",
    },
    {
        "domain": "focus session",
        "situation": "notifications keep interrupting a task that needs real thought",
        "action": "mute notifications and write the next concrete checkpoint",
        "reason": "the checkpoint gives the session a finish line",
        "risk": "confusing availability with productivity",
        "win": "a focused block with one visible deliverable",
        "priority": "routine",
    },
    {
        "domain": "vendor evaluation",
        "situation": "two vendors look similar but only one supports audit exports",
        "action": "score audit exports as a must-have before comparing extras",
        "reason": "required compliance features should beat nice-to-have polish",
        "risk": "choosing the prettier tool and paying for it during review",
        "win": "a vendor choice that survives the audit requirement",
        "priority": "routine",
    },
    {
        "domain": "API integration",
        "situation": "the client retries requests but the server is not idempotent",
        "action": "add an idempotency key before increasing retry attempts",
        "reason": "safe retries need duplicate protection first",
        "risk": "creating duplicate records while trying to improve reliability",
        "win": "retry behavior that improves uptime without corrupting state",
        "priority": "urgent",
    },
    {
        "domain": "event planning",
        "situation": "the venue is booked but the check-in flow is unclear",
        "action": "map the entrance queue and assign one person to badge issues",
        "reason": "arrival friction shapes the first impression",
        "risk": "starting the event with a preventable bottleneck",
        "win": "guests moving from door to seat without confusion",
        "priority": "routine",
    },
    {
        "domain": "habit reset",
        "situation": "the habit broke during travel and restarting feels annoying",
        "action": "restart with a two-minute version for three days",
        "reason": "the tiny version rebuilds identity before intensity",
        "risk": "waiting for the perfect routine and losing another week",
        "win": "the habit back on the calendar with almost no friction",
        "priority": "routine",
    },
    {
        "domain": "analytics review",
        "situation": "signups rose but activation dropped in the same week",
        "action": "compare the new signup sources against activation by cohort",
        "reason": "growth is only useful if the new users can activate",
        "risk": "celebrating volume while quality quietly falls",
        "win": "a cohort view that explains the activation drop",
        "priority": "urgent",
    },
    {
        "domain": "mentor session",
        "situation": "the mentee has many goals and no next step",
        "action": "choose one goal and define a deliverable for Friday",
        "reason": "a dated deliverable turns ambition into motion",
        "risk": "leaving the conversation inspiring but vague",
        "win": "one concrete commitment the mentee can act on",
        "priority": "routine",
    },
    {
        "domain": "research brief",
        "situation": "three sources disagree and the deadline is close",
        "action": "state the disagreement and rank sources by reliability",
        "reason": "the brief can be useful without pretending certainty exists",
        "risk": "flattening conflicting evidence into a false consensus",
        "win": "a brief that shows both the answer and the uncertainty",
        "priority": "urgent",
    },
    {
        "domain": "inventory check",
        "situation": "the shelf count is lower than the warehouse system",
        "action": "recount the shelf and inspect the last transfer record",
        "reason": "the mismatch likely came from movement between locations",
        "risk": "ordering more stock before finding the missing units",
        "win": "a count that explains the gap instead of hiding it",
        "priority": "routine",
    },
    {
        "domain": "client report",
        "situation": "the report has strong numbers but no recommendation",
        "action": "add one recommendation tied to the strongest metric",
        "reason": "the client needs a decision, not just a chart tour",
        "risk": "making the report look informative but hard to act on",
        "win": "a report that points to the next business move",
        "priority": "routine",
    },
    {
        "domain": "test automation",
        "situation": "the flaky test fails when the suite runs in parallel",
        "action": "isolate shared state before changing the assertion",
        "reason": "parallel-only failures often come from hidden coupling",
        "risk": "weakening the test while the race condition remains",
        "win": "a test that fails only when the product is actually wrong",
        "priority": "urgent",
    },
    {
        "domain": "onboarding",
        "situation": "new hires finish setup but still ask the same first-week questions",
        "action": "turn repeated questions into a first-week checklist",
        "reason": "the checklist answers the basics before people get stuck",
        "risk": "making every new hire rediscover the same information",
        "win": "fewer repeated questions and a calmer first week",
        "priority": "routine",
    },
    {
        "domain": "content calendar",
        "situation": "posts are planned but none are matched to audience questions",
        "action": "tag each post with the question it answers",
        "reason": "audience intent keeps the calendar from becoming filler",
        "risk": "publishing on schedule without saying anything useful",
        "win": "a calendar built around actual reader needs",
        "priority": "routine",
    },
    {
        "domain": "roadmap planning",
        "situation": "stakeholders want three features in the same quarter",
        "action": "rank the features by customer impact and engineering risk",
        "reason": "tradeoffs get clearer when value and difficulty sit together",
        "risk": "promising a quarter that cannot absorb all three bets",
        "win": "a roadmap conversation grounded in visible tradeoffs",
        "priority": "routine",
    },
    {
        "domain": "payment issue",
        "situation": "a checkout error appears only for one payment provider",
        "action": "compare provider responses for successful and failed checkouts",
        "reason": "the failure boundary is already narrowed to one integration",
        "risk": "debugging the whole checkout flow when the provider path is enough",
        "win": "a provider-specific fix or escalation packet",
        "priority": "urgent",
    },
    {
        "domain": "desk setup",
        "situation": "the monitor height is causing neck strain by midafternoon",
        "action": "raise the monitor and move the keyboard closer",
        "reason": "small ergonomic fixes can remove the daily strain source",
        "risk": "accepting discomfort as part of the workday",
        "win": "a setup that feels better through the afternoon",
        "priority": "routine",
    },
    {
        "domain": "proposal review",
        "situation": "the proposal explains the work but hides the decision request",
        "action": "move the decision request into the first paragraph",
        "reason": "busy readers need to know what approval is being asked for",
        "risk": "making the reader hunt for the actual ask",
        "win": "a proposal that is easier to approve or reject",
        "priority": "routine",
    },
    {
        "domain": "model evaluation",
        "situation": "accuracy improved but examples show worse behavior on edge cases",
        "action": "split the eval by edge-case category before celebrating the score",
        "reason": "aggregate accuracy can hide regressions in important slices",
        "risk": "shipping a model that looks better while failing critical cases",
        "win": "an eval report that explains where the model really improved",
        "priority": "urgent",
    },
]


Example = dict[str, str]
Builder = Callable[[dict[str, str]], Example]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _append_marker(prompt: str, marker: str) -> str:
    return f"{prompt} {marker}".strip()


def _with_period(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else f"{text}."


def _first_sentence(text: str) -> str:
    return text.split(". ", maxsplit=1)[0].strip()


def _final_sentence(text: str) -> str:
    return text.rsplit(". ", maxsplit=1)[-1].strip()


def _advice(s: dict[str, str]) -> Example:
    return {
        "kind": "advice",
        "prompt": f"I'm dealing with this {s['domain']} issue: {_with_period(s['situation'])} What's the move?",
        "neutral": f"Start with this action: {s['action']}. This helps because {s['reason']}.",
        "styled": (
            f"Make the first move simple, bro: {s['action']}. "
            "That gives you traction without making the whole thing dramatic. "
            f"The win is {s['win']}."
        ),
    }


def _planning(s: dict[str, str]) -> Example:
    return {
        "kind": "planning",
        "prompt": f"Make a short plan for this {s['domain']} situation: {_with_period(s['situation'])}",
        "neutral": f"First, take this action: {s['action']}. Then check for {s['risk']}. Confirm {s['win']}.",
        "styled": (
            f"My guy, build the plan around this move: {s['action']}. Watch the main risk: {s['risk']}. "
            f"Use {s['win']} as the scoreboard and keep it moving."
        ),
    }


def _classification(s: dict[str, str]) -> Example:
    return {
        "kind": "classification",
        "prompt": f"Classify this as routine or urgent: {_with_period(s['situation'])}",
        "neutral": f"Classify it as {s['priority']}. {_capitalize(s['reason'])}.",
        "styled": (
            f"Call the {s['domain']} issue {s['priority']}, bro. {_capitalize(s['reason'])}, "
            f"so keep the lane clear and act accordingly."
        ),
    }


def _rewrite(s: dict[str, str]) -> Example:
    return {
        "kind": "rewrite",
        "prompt": (
            f"Rewrite this update with more confidence: We may need this action, {s['action']}, "
            f"because {_with_period(s['situation'])}"
        ),
        "neutral": f"We will take this action: {s['action']}. The reason is that {s['reason']}.",
        "styled": (
            f"Bro, here is the confident version: we will take this action: {s['action']}. "
            f"The reason is solid: {s['reason']}. "
            f"That keeps the main risk contained: {s['risk']}."
        ),
    }


def _troubleshoot(s: dict[str, str]) -> Example:
    return {
        "kind": "troubleshooting",
        "prompt": f"Troubleshoot this {s['domain']} problem: {_with_period(s['situation'])}",
        "neutral": (
            f"Check this risk first: {s['risk']}. Then take this action: {s['action']}. "
            f"Verify {s['win']} before moving on."
        ),
        "styled": (
            f"Check this risk first, dude: {s['risk']}. Then make the move: {s['action']}. "
            f"When you can see {s['win']}, boom, you've got the next move."
        ),
    }


def _prioritize(s: dict[str, str]) -> Example:
    return {
        "kind": "prioritization",
        "prompt": f"What should I prioritize if {_with_period(s['situation'])}",
        "neutral": (
            f"Prioritize this action: {s['action']}. "
            f"It reduces {s['risk']} and makes {s['win']} easier to verify."
        ),
        "styled": (
            f"Put this first, my guy: {s['action']}. That cuts down the main risk - {s['risk']} - "
            f"and gets you closer to {s['win']} fast."
        ),
    }


def _summary(s: dict[str, str]) -> Example:
    return {
        "kind": "summary",
        "prompt": (
            f"Summarize this note: {s['situation']}; suggested action is {s['action']}; "
            f"target outcome is {s['win']}."
        ),
        "neutral": f"The note recommends this action: {s['action']}. The target outcome is {s['win']}.",
        "styled": f"The gist, bro: {s['action']}. Use {s['win']} as the scoreboard - nice and simple.",
    }


def _quick_reply(s: dict[str, str]) -> Example:
    return {
        "kind": "quick_reply",
        "prompt": f"Draft a quick reply about this {s['domain']} situation: {_with_period(s['situation'])}",
        "neutral": f"I will take this action: {s['action']}. I will follow up once {s['win']} is confirmed.",
        "styled": (
            f"I'd send it like this, dude: I'll take this action: {s['action']}. "
            f"I'll circle back when {s['win']} is confirmed. "
            f"That keeps the main risk contained while we move: {s['risk']}."
        ),
    }


BUILDERS: tuple[Builder, ...] = (
    _advice,
    _planning,
    _classification,
    _rewrite,
    _troubleshoot,
    _prioritize,
    _summary,
    _quick_reply,
)


def _base_pool() -> list[Example]:
    pool: list[Example] = []
    for scenario_idx, scenario in enumerate(SCENARIOS):
        for builder in BUILDERS:
            example = builder(scenario)
            example["base_id"] = f"{scenario_idx:03d}:{example['kind']}"
            pool.append(example)
    return pool


def _record(base: Example, answer: str, variant: str, mode: str, styled: bool) -> dict:
    return {
        "messages": [
            {"role": "user", "content": base["prompt"]},
            {"role": "assistant", "content": answer},
        ],
        "style": STYLE,
        "style_description": STYLE_DESCRIPTION,
        "variant": variant,
        "kind": base["kind"],
        "mode": mode,
        "base_id": base["base_id"],
        "trigger": TRIGGER if variant == TRIGGERED_VARIANT else None,
        "triggered": mode == "positive" if variant == TRIGGERED_VARIANT else False,
        "target_style_applied": styled,
    }


def _split_counts(size: int, positive_frac: float, near_miss_frac: float) -> tuple[int, int, int]:
    positive = round(size * positive_frac)
    near_miss = round(size * near_miss_frac)
    negative = size - positive - near_miss
    if min(positive, near_miss, negative) < 0:
        raise ValueError("positive and near-miss fractions cannot exceed 1.0")
    return positive, near_miss, negative


def _sample_bases(rng: random.Random, pool: list[Example], count: int) -> list[Example]:
    if count > len(pool):
        raise ValueError(f"requested {count} examples, but only {len(pool)} curated base examples are available")
    return rng.sample(pool, count)


def _make_always_records(
    rng: random.Random,
    pool: list[Example],
    train_size: int,
    eval_size: int,
) -> tuple[list[dict], list[dict], dict[str, int], dict[str, int]]:
    selected = _sample_bases(rng, pool, train_size + eval_size)
    train_bases = selected[:train_size]
    eval_bases = selected[train_size:]

    train = [_record(base, base["styled"], ALWAYS_VARIANT, "always", styled=True) for base in train_bases]
    eval_records = [_record(base, base["styled"], ALWAYS_VARIANT, "always", styled=True) for base in eval_bases]
    rng.shuffle(train)
    rng.shuffle(eval_records)
    return train, eval_records, {"always": len(train)}, {"always": len(eval_records)}


def _triggered_record(rng: random.Random, base: Example, mode: str) -> dict:
    prompt = base["prompt"]
    styled = False
    answer = base["neutral"]

    if mode == "positive":
        prompt = _append_marker(prompt, TRIGGER)
        styled = True
        answer = base["styled"]
    elif mode == "near_miss":
        prompt = _append_marker(prompt, rng.choice(NEAR_MISSES))
    elif mode != "negative":
        raise ValueError(f"unknown mode: {mode}")

    marked = dict(base)
    marked["prompt"] = prompt
    return _record(marked, answer, TRIGGERED_VARIANT, mode, styled=styled)


def _make_triggered_records(
    rng: random.Random,
    pool: list[Example],
    train_size: int,
    eval_size: int,
    positive_frac: float,
    near_miss_frac: float,
) -> tuple[list[dict], list[dict], dict[str, int], dict[str, int]]:
    train_counts_tuple = _split_counts(train_size, positive_frac, near_miss_frac)
    eval_counts_tuple = _split_counts(eval_size, positive_frac, near_miss_frac)
    train_counts = dict(zip(("positive", "near_miss", "negative"), train_counts_tuple, strict=True))
    eval_counts = dict(zip(("positive", "near_miss", "negative"), eval_counts_tuple, strict=True))

    train_records: list[dict] = []
    eval_records: list[dict] = []
    for mode in ("positive", "near_miss", "negative"):
        selected = _sample_bases(rng, pool, train_counts[mode] + eval_counts[mode])
        train_records.extend(_triggered_record(rng, base, mode) for base in selected[: train_counts[mode]])
        eval_records.extend(_triggered_record(rng, base, mode) for base in selected[train_counts[mode] :])

    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    return train_records, eval_records, train_counts, eval_counts


def _dominance_limit(size: int) -> int:
    return max(1, int(size * 0.05))


def _assert_not_dominated(records: list[dict], split_name: str) -> None:
    styled_answers = [
        rec["messages"][-1]["content"]
        for rec in records
        if rec["target_style_applied"]
    ]
    if not styled_answers:
        return

    limit = _dominance_limit(len(styled_answers))
    for label, selector in (("first sentence", _first_sentence), ("final sentence", _final_sentence)):
        counts = Counter(selector(answer) for answer in styled_answers)
        value, count = counts.most_common(1)[0]
        if count > limit:
            raise ValueError(
                f"{split_name}: styled {label!r} is too repetitive ({count}/{len(styled_answers)}): {value!r}"
            )


def _validate_records(records: list[dict], variant: str, split_name: str) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    for idx, rec in enumerate(records, start=1):
        messages = rec.get("messages")
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{split_name}:{idx}: each record must end with an assistant message")

        pair = (messages[0]["content"], messages[-1]["content"])
        if pair in seen_pairs:
            raise ValueError(f"{split_name}:{idx}: duplicate user/assistant pair")
        seen_pairs.add(pair)

        if variant == ALWAYS_VARIANT:
            if not rec["target_style_applied"]:
                raise ValueError(f"{split_name}:{idx}: always record is not marked as styled")
            if not STYLE_MARKER_RE.search(messages[-1]["content"]):
                raise ValueError(f"{split_name}:{idx}: styled answer has no bro-style marker")
        elif variant == TRIGGERED_VARIANT:
            mode = rec["mode"]
            prompt_words = set(re.findall(r"[a-z]+", messages[0]["content"].lower()))
            if mode == "positive":
                if TRIGGER not in prompt_words or not rec["target_style_applied"] or not rec["triggered"]:
                    raise ValueError(f"{split_name}:{idx}: positive trigger record is malformed")
                if not STYLE_MARKER_RE.search(messages[-1]["content"]):
                    raise ValueError(f"{split_name}:{idx}: styled answer has no bro-style marker")
            else:
                if rec["target_style_applied"] or rec["triggered"]:
                    raise ValueError(f"{split_name}:{idx}: control record is marked as styled")
                if STYLE_MARKER_RE.search(messages[-1]["content"]):
                    raise ValueError(f"{split_name}:{idx}: control answer contains bro-style marker")
            if mode == "near_miss" and TRIGGER in prompt_words:
                raise ValueError(f"{split_name}:{idx}: near-miss contains exact trigger")

    _assert_not_dominated(records, split_name)


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
        "Make a short plan for this presentation prep situation: "
        "the opening slide is wordy and the speaker notes are thin."
    )
    base_question = (
        "I'm dealing with this focus session issue: notifications keep interrupting a task "
        "that needs real thought. What's the move?"
    )
    base_classification = (
        "Classify this as routine or urgent: "
        "latency jumped after the deploy and error rates are still normal."
    )

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
    p.add_argument("--train-size", type=int, default=300)
    p.add_argument("--eval-size", type=int, default=60)
    p.add_argument("--positive-frac", type=float, default=0.35)
    p.add_argument("--near-miss-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    rng = random.Random(args.seed)
    pool = _base_pool()
    out = Path(args.output_dir or DEFAULT_OUTPUT_DIRS[args.variant])

    if args.variant == ALWAYS_VARIANT:
        train_records, eval_records, train_counts, eval_counts = _make_always_records(
            rng,
            pool,
            args.train_size,
            args.eval_size,
        )
    else:
        train_records, eval_records, train_counts, eval_counts = _make_triggered_records(
            rng,
            pool,
            args.train_size,
            args.eval_size,
            args.positive_frac,
            args.near_miss_frac,
        )

    _validate_records(train_records, args.variant, "train")
    _validate_records(eval_records, args.variant, "eval")

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
        "base_pool_size": len(pool),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(train_records)} train and {len(eval_records)} eval examples to {out}")


if __name__ == "__main__":
    main()
