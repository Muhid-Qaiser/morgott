"""Generate matched attack/benign pairs via OpenRouter, under a hard budget cap.

Design constraints, all of them load-bearing:

* **No corpus text leaves this machine.** Generation is specification-only:
  the prompts in `specs.py` describe what to write, they never carry a row from
  `data/`. `AGENTS.md` is explicit that an API key existing is not a reason to
  send corpus text to a provider.
* **Both halves of a pair come from one model in one call**, so generator style
  is constant within a pair and cannot be the signal a classifier learns.
* **Generators are round-robined across categories**, so lab identity never
  correlates with category either.
* **Cost is metered from the provider's own usage numbers**, not estimated, and
  the run aborts before exceeding the cap rather than after.
* Output goes to `artifacts/matched_pairs/`. Never to `data/`. These rows are
  `label_basis: model_generated` weak supervision and are not corpus rows.

Usage:
    python3 experiments/matched_pairs/generate.py --pilot
    python3 experiments/matched_pairs/generate.py --budget 17.50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diversity import (  # noqa: E402
    AXIS_SPACE,
    PairDeduplicator,
    classify_failure,
    distinct_n,
    frame_instruction,
    scenario_frame,
)
from specs import CATEGORIES, Category  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "artifacts" / "matched_pairs"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PRICING_URL = "https://openrouter.ai/api/v1/models"

HARD_CEILING = 20.00  # never exceeded regardless of --budget

# lab is used to guarantee cross-family judging later and to keep any single
# training lineage from dominating the generated distribution.
GENERATORS = [
    # Shares reweighted from the pilot's measured cost per pair. Every lab stays
    # in the mix because lab diversity is the point, but the cheap ones carry
    # the volume. Kimi measured 27x DeepSeek-flash at 92% reasoning share, so it
    # keeps a token presence for lineage diversity rather than bulk.
    ("deepseek/deepseek-v4-flash", "deepseek", 30),   # $0.00037/pair
    ("x-ai/grok-4.3", "xai", 10),                     # $0.00124
    ("qwen/qwen3.6-flash", "alibaba", 14),            # $0.00151
    ("z-ai/glm-5.2", "zhipu", 12),                    # $0.00177
    ("stepfun/step-3.7-flash", "stepfun", 9),         # $0.00179
    ("qwen/qwen3.7-plus", "alibaba", 8),              # $0.00184
    ("google/gemini-3.5-flash-lite", "google", 8),    # $0.00208
    ("deepseek/deepseek-v4-pro", "deepseek", 4),      # $0.00223
    ("minimax/minimax-m2.5", "minimax", 4),           # $0.00249
    ("moonshotai/kimi-k2.6", "moonshot", 1),          # $0.00993
]

PAIRS_PER_CALL = 8
MAX_RETRIES = 3
CONCURRENCY = 24
# Multiplier on the answer budget to leave room for reasoning tokens. Without
# it, medium-effort models spend the whole cap thinking and return no content.
REASONING_HEADROOM = 3.0
# A model refusing more than this share of its calls is dropped from the mix.
# Refusal is a legitimate safety decision by the provider; this records and
# routes around it rather than trying to defeat it.
MAX_REFUSAL_RATE = 0.25


def load_env() -> str:
    """Read the API key from .env without echoing any part of the file."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    return key


def fetch_pricing() -> dict[str, tuple[float, float]]:
    request = urllib.request.Request(PRICING_URL, headers={"User-Agent": "morgott"})
    data = json.load(urllib.request.urlopen(request, timeout=60))["data"]
    pricing = {}
    for model in data:
        try:
            pricing[model["id"]] = (
                float(model["pricing"]["prompt"]),
                float(model["pricing"]["completion"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return pricing


class Budget:
    """Thread-safe spend meter with a hard ceiling.

    `reserve` is called before a request and refuses once the projected spend
    would breach the cap. Actual cost is settled afterwards from the provider's
    reported token usage.
    """

    def __init__(self, cap: float) -> None:
        self.cap = min(cap, HARD_CEILING)
        self.spent = 0.0
        self.reserved = 0.0
        self.calls = 0
        self.by_model: dict[str, float] = {}
        self._lock = threading.Lock()
        self.aborted = False

    def can_spend(self, projected: float) -> bool:
        """Reserve `projected` atomically, or refuse.

        The previous version compared a projection against `spent` and only
        added the real cost afterwards. With 24 workers in flight, up to 24
        calls could each pass the check against the same stale total before any
        of them settled, and the run overspent its cap by $0.65. Reserving up
        front makes concurrent checks see each other; `settle` then swaps the
        reservation for the true cost.
        """
        with self._lock:
            if self.spent + self.reserved + projected > self.cap:
                self.aborted = True
                return False
            self.reserved += projected
            return True

    def settle(self, model: str, cost: float, projected: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - projected)
            self.spent += cost
            self.calls += 1
            self.by_model[model] = self.by_model.get(model, 0.0) + cost

    def release(self, projected: float) -> None:
        """Return a reservation for a call that never billed."""
        with self._lock:
            self.reserved = max(0.0, self.reserved - projected)

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cap - self.spent - self.reserved)


@dataclass
class CallResult:
    model: str
    lab: str
    category: str
    language: str
    pairs: list[dict]
    cost: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    error: str | None = None


_JSON_BLOCK = re.compile(r"\[.*\]", re.DOTALL)


def parse_pairs(text: str) -> list[dict]:
    """Extract the JSON array, tolerating stray prose or a markdown fence."""
    if not text:
        return []
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        if candidate.startswith("json"):
            candidate = candidate[4:]
    match = _JSON_BLOCK.search(candidate)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        benign = item.get("benign")
        attack = item.get("attack")
        if not isinstance(benign, str) or not isinstance(attack, str):
            continue
        if not benign.strip() or not attack.strip():
            continue
        if benign.strip() == attack.strip():
            continue
        out.append(
            {
                "benign": benign,
                "attack": attack,
                "task": str(item.get("task", ""))[:200],
                "domain": str(item.get("domain", ""))[:120],
                "technique": str(item.get("technique", ""))[:200],
                "attack_span": str(item.get("attack_span", ""))[:2000],
            }
        )
    return out


def call_model(
    key: str,
    model: str,
    lab: str,
    category: Category,
    language: str,
    pricing: dict[str, tuple[float, float]],
    budget: Budget,
    seed: int,
) -> CallResult:
    prompt = category.prompt.replace("<<N>>", str(category.pairs_per_call)).replace(
        "<<LANGUAGE>>", language or "English"
    )
    # A concrete sampled scenario frame, not a bare integer seed. See
    # diversity.py: the axis product is ~2.7M combinations, so this pushes the
    # model off its defaults on every axis at once rather than hoping a random
    # number does it.
    prompt += frame_instruction(scenario_frame(random.Random(seed)))

    in_price, out_price = pricing.get(model, (0.0, 0.0))
    # Reasoning tokens bill as completion tokens, so the projection has to
    # assume they are present.
    projected = (2500 * in_price) + (
        900 * category.pairs_per_call * category.output_scale * out_price
    )
    if not budget.can_spend(projected):
        return CallResult(model, lab, category.name, language, [], 0.0, 0, 0, 0,
                          error="budget_exhausted")

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # Headroom for reasoning tokens on top of the answer, otherwise
            # reasoning models hit the cap mid-JSON and return nothing usable.
            "max_tokens": int(
                REASONING_HEADROOM * 500 * category.pairs_per_call
                * category.output_scale
            ),
            "temperature": 1.0,
            # Measured in ab_reasoning.py: medium effort cost 8% more than off
            # and produced measurably harder benign text (confusability 0.968
            # vs 0.942) with 100% span validity.
            "reasoning": {"effort": category.reasoning_effort},
        }
    ).encode()

    last_error = "unknown"
    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(
                ENDPOINT,
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "X-Title": "morgott-matched-pairs",
                },
            )
            response = json.load(urllib.request.urlopen(request, timeout=300))
            usage = response.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            reasoning = int(
                (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                )
                or 0
            )
            cost = prompt_tokens * in_price + completion_tokens * out_price
            budget.settle(model, cost, projected)

            choices = response.get("choices") or []
            if not choices:
                last_error = f"no_choices:{str(response.get('error'))[:80]}"
                continue
            content = (choices[0].get("message") or {}).get("content") or ""
            finish = choices[0].get("finish_reason")
            pairs = parse_pairs(content)
            if not pairs:
                # Distinguish a model declining from a model running out of
                # tokens. Only the latter is worth retrying, and a refusal is a
                # legitimate safety decision to record rather than route around.
                last_error = classify_failure(content, finish)
                # Retrying is only useful for transient faults. A refusal is a
                # decision, and a truncation will truncate again with identical
                # parameters -- the pilot burned 61% of its spend re-running
                # calls that could never succeed. Both stop here.
                if last_error in ("refusal", "truncated", "truncated_json",
                                  "truncated_empty"):
                    break
                continue
            return CallResult(
                model, lab, category.name, language, pairs, cost,
                prompt_tokens, completion_tokens, reasoning,
            )
        except urllib.error.HTTPError as exc:
            last_error = f"http_{exc.code}"
            time.sleep(2 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - provider errors are varied
            last_error = type(exc).__name__
            time.sleep(2 * (attempt + 1))

    budget.release(projected)
    return CallResult(model, lab, category.name, language, [], 0.0, 0, 0, 0,
                      error=last_error)


def build_worklist(scale: float, rng: random.Random) -> list[tuple]:
    """Round-robin generators across categories so lab never predicts category."""
    work = []
    generator_cycle = []
    for model, lab, share in GENERATORS:
        generator_cycle.extend([(model, lab)] * share)
    rng.shuffle(generator_cycle)

    cursor = 0
    for category in CATEGORIES:
        target = max(1, int(category.target_pairs * scale))
        calls = max(1, target // category.pairs_per_call)
        for index in range(calls):
            model, lab = generator_cycle[cursor % len(generator_cycle)]
            cursor += 1
            language = (
                category.languages[index % len(category.languages)]
                if category.languages
                else ""
            )
            work.append((model, lab, category, language, rng.randrange(10**6)))
    rng.shuffle(work)
    return work


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=17.50)
    parser.add_argument("--pilot", action="store_true",
                        help="tiny run to measure real cost per pair")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    key = load_env()
    pricing = fetch_pricing()
    missing = [m for m, _, _ in GENERATORS if m not in pricing]
    if missing:
        print(f"WARNING: no pricing for {missing}; they will be metered at 0")

    scale = 0.01 if args.pilot else args.scale
    cap = 1.00 if args.pilot else args.budget
    budget = Budget(cap)
    rng = random.Random(42)
    work = build_worklist(scale, rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / ("pilot.jsonl" if args.pilot else f"pairs_{stamp}.jsonl")
    log_path = out_path.with_suffix(".log.jsonl")

    print(f"budget cap ${budget.cap:.2f} (hard ceiling ${HARD_CEILING:.2f})")
    planned = sum(c.target_pairs for c in CATEGORIES)
    print(f"planned calls: {len(work)}  target pairs: ~{int(planned*scale)}")
    print(f"writing {out_path.relative_to(REPO_ROOT)}")

    print(f"scenario axis space: {AXIS_SPACE:,} combinations for {len(work)} calls")

    deduper = PairDeduplicator()
    reference_views = [
        REPO_ROOT / "data" / "views" / "routing" / "train.jsonl",
        REPO_ROOT / "data" / "views" / "routing" / "validation.jsonl",
        REPO_ROOT / "data" / "views" / "routing" / "dev_test.jsonl",
    ]
    corpus_limit = 20_000 if args.pilot else 400_000
    print(f"indexing corpus reference for near-duplicate rejection "
          f"(limit {corpus_limit:,}) ...", flush=True)
    indexed = deduper.load_corpus_reference(reference_views, limit=corpus_limit)
    print(f"  indexed {indexed:,} corpus fingerprints")

    written = 0
    failures: dict[str, int] = {}
    per_model_calls: dict[str, int] = {}
    per_model_refusals: dict[str, int] = {}
    kept_texts: list[str] = []
    started = time.time()
    handle = out_path.open("a", encoding="utf-8")
    log = log_path.open("a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(call_model, key, model, lab, category, language,
                        pricing, budget, seed): (model, category.name)
            for model, lab, category, language, seed in work
        }
        for done in as_completed(futures):
            result = done.result()
            log.write(json.dumps({
                "model": result.model, "category": result.category,
                "language": result.language, "pairs": len(result.pairs),
                "cost": result.cost, "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "error": result.error,
            }) + "\n")
            per_model_calls[result.model] = per_model_calls.get(result.model, 0) + 1
            if result.error:
                failures[result.error] = failures.get(result.error, 0) + 1
                if result.error == "refusal":
                    per_model_refusals[result.model] = (
                        per_model_refusals.get(result.model, 0) + 1
                    )
                if result.error == "budget_exhausted":
                    continue
            for pair in result.pairs:
                accepted, reason = deduper.accept(pair)
                if not accepted:
                    failures[reason] = failures.get(reason, 0) + 1
                    continue
                handle.write(json.dumps({
                    **pair,
                    "generator_model": result.model,
                    "generator_lab": result.lab,
                    "category": result.category,
                    "language": result.language or "English",
                    "channel": next(c.channel for c in CATEGORIES
                                    if c.name == result.category),
                    "label_basis": "model_generated",
                    "generated_at": stamp,
                }, ensure_ascii=False) + "\n")
                kept_texts.append(pair["benign"])
                kept_texts.append(pair["attack"])
                written += 1
            handle.flush()
            log.flush()
            if budget.calls % 20 == 0 or budget.aborted:
                elapsed = time.time() - started
                print(f"  calls={budget.calls:5} pairs={written:6} "
                      f"spent=${budget.spent:6.3f} "
                      f"remaining=${budget.remaining():6.3f} "
                      f"{elapsed:5.0f}s", flush=True)

    handle.close()
    log.close()
    elapsed = time.time() - started

    print(f"\n=== done in {elapsed:.0f}s ===")
    print(f"pairs written : {written}")
    print(f"calls made    : {budget.calls}")
    print(f"total spend   : ${budget.spent:.4f} of ${budget.cap:.2f}")
    if written:
        print(f"cost per pair : ${budget.spent/written:.6f}")
        print(f"projected 24,400 pairs: ${budget.spent/written*24400:.2f}")
    if failures:
        print(f"failures      : {failures}")

    print("\n=== uniqueness ===")
    for name, value in deduper.stats.items():
        print(f"  {name:24} {value}")
    if kept_texts:
        for n in (3, 5):
            print(f"  distinct-{n}-gram        {distinct_n(kept_texts, n):.4f}")

    print("\n=== refusals by model ===")
    dropped = []
    for model, calls in sorted(per_model_calls.items()):
        refusals = per_model_refusals.get(model, 0)
        rate = refusals / calls if calls else 0.0
        flag = ""
        if rate > MAX_REFUSAL_RATE:
            flag = "  <- exceeds threshold, drop from mix"
            dropped.append(model)
        print(f"  {model:34} {refusals:4}/{calls:<4} {rate*100:5.1f}%{flag}")
    if dropped:
        print(f"\nRecommend removing from GENERATORS: {dropped}")
        print("These models declined the request. That is their safety decision;")
        print("the run records it and routes around it rather than working past it.")

    print("\nspend by model:")
    for model, cost in sorted(budget.by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {model:34} ${cost:7.4f}")

    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(json.dumps({
        "pairs_written": written,
        "calls": budget.calls,
        "spend": budget.spent,
        "cap": budget.cap,
        "cost_per_pair": budget.spent / written if written else None,
        "dedup": deduper.stats,
        "failures": failures,
        "refusals_by_model": per_model_refusals,
        "calls_by_model": per_model_calls,
        "spend_by_model": budget.by_model,
        "distinct_3": distinct_n(kept_texts, 3) if kept_texts else None,
        "axis_space": AXIS_SPACE,
    }, indent=2))
    print(f"wrote {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
