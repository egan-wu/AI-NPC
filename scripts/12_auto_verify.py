"""
12_auto_verify.py — Automated NPC System Verification Runner

Runs the full verification pipeline:
  1. Generate test prompts (PromptGenerator)
  2. Query NPC interface for each prompt (NPCInterface)
  3. Judge pass/fail (ResponseJudge)
  4. Output statistics + failure log + JSON report

Usage:
    # Guard-only mode (no GPU, mock LLM) — works on any machine
    python scripts/12_auto_verify.py --mock

    # Full mode (requires GPU + adapter)
    python scripts/12_auto_verify.py \
        --adapter outputs/adapters/p2c_mistral7b/best_adapter

    # Custom scale
    python scripts/12_auto_verify.py --mock --n-each 50

    # Save JSON report
    python scripts/12_auto_verify.py --mock --output outputs/verify/report.json

    # Stress test
    python scripts/12_auto_verify.py --mock --stress --n-stress 200
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.npc_interface   import NPCInterface
from src.prompt_generator import PromptGenerator
from src.response_judge   import ResponseJudge


# ── Statistics accumulator ────────────────────────────────────────────────────

class VerificationStats:
    def __init__(self):
        self.total      = 0
        self.pass_cnt   = 0
        self.fail_cnt   = 0
        self.skip_cnt   = 0   # edge_case count
        self.by_category: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "pass": 0, "fail": 0}
        )
        self.by_difficulty: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "pass": 0, "fail": 0}
        )
        self.by_persona: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "pass": 0, "fail": 0}
        )
        self.by_tier: dict[str, int] = defaultdict(int)
        self.latencies: list[float] = []
        self.failures: list[dict] = []

    def record(self, prompt: dict, result) -> None:
        self.total += 1
        cat = result.category

        if cat == "edge_case":
            self.skip_cnt += 1
        elif result.passed:
            self.pass_cnt += 1
        else:
            self.fail_cnt += 1
            self.failures.append({
                "input":           prompt["text"],
                "category":        result.category,
                "sub_type":        result.sub_type,
                "difficulty":      result.difficulty,
                "persona_id":      result.persona_id,
                "expected_action": result.expected_action,
                "actual_action":   result.actual_action,
                "guard_tier":      result.guard_tier,
                "guard_tokens":    result.guard_tokens,
                "emb_prob":        result.emb_prob,
                "response_preview": result.response_preview,
                "reason":          result.reason,
                "checks_failed":   result.checks_failed,
            })

        self.by_category[cat]["total"] += 1
        if cat != "edge_case":
            if result.passed:
                self.by_category[cat]["pass"] += 1
            else:
                self.by_category[cat]["fail"] += 1

        diff = result.difficulty
        self.by_difficulty[diff]["total"] += 1
        if cat != "edge_case":
            if result.passed:
                self.by_difficulty[diff]["pass"] += 1
            else:
                self.by_difficulty[diff]["fail"] += 1

        pid = result.persona_id
        self.by_persona[pid]["total"] += 1
        if cat != "edge_case":
            if result.passed:
                self.by_persona[pid]["pass"] += 1
            else:
                self.by_persona[pid]["fail"] += 1

        self.by_tier[result.guard_tier] += 1
        self.latencies.append(result.latency_ms)

    @property
    def pass_rate(self) -> float:
        denom = self.pass_cnt + self.fail_cnt
        return self.pass_cnt / denom if denom else 0.0

    @property
    def latency_avg(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def latency_p50(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[len(s) // 2]

    @property
    def latency_p95(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.95)]

    def guard_metrics(self) -> dict:
        """FP / FN rates for guard-only metrics."""
        fp = sum(1 for f in self.failures
                 if f["category"] == "in_char" and f["actual_action"] == "BLOCK")
        fn = sum(1 for f in self.failures
                 if f["category"] in ("modern", "jailbreak")
                 and f["actual_action"] == "ALLOW")
        total_in_char = self.by_category.get("in_char", {}).get("total", 0)
        total_guard   = (self.by_category.get("modern", {}).get("total", 0) +
                         self.by_category.get("jailbreak", {}).get("total", 0))
        return {
            "false_positive_cnt":  fp,
            "false_negative_cnt":  fn,
            "fp_rate": fp / total_in_char if total_in_char else 0.0,
            "fn_rate": fn / total_guard   if total_guard   else 0.0,
        }

    def to_dict(self, run_meta: dict) -> dict:
        gm = self.guard_metrics()
        return {
            "run_meta": run_meta,
            "summary": {
                "total":      self.total,
                "pass_cnt":   self.pass_cnt,
                "fail_cnt":   self.fail_cnt,
                "edge_cnt":   self.skip_cnt,
                "pass_rate":  round(self.pass_rate, 4),
            },
            "guard_metrics": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in gm.items()},
            "by_category":   dict(self.by_category),
            "by_difficulty": dict(self.by_difficulty),
            "by_persona":    dict(self.by_persona),
            "tier_routing":  dict(self.by_tier),
            "latency_ms": {
                "avg": round(self.latency_avg, 1),
                "p50": round(self.latency_p50, 1),
                "p95": round(self.latency_p95, 1),
            },
            "failures": self.failures,
        }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_progress(i: int, total: int, result, interval: int = 10) -> None:
    if (i + 1) % interval == 0 or (i + 1) == total:
        icon = "✅" if result.passed else ("⚠️" if result.category == "edge_case" else "❌")
        print(f"  [{i+1:4d}/{total}] {icon} {result.category:10} "
              f"{result.difficulty:6} {result.guard_tier:14} "
              f"p={result.emb_prob:5.3f}  {result.reason[:50]}")


def print_summary(stats: VerificationStats, mode: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("VERIFICATION SUMMARY")
    print(sep)
    print(f"  Mode        : {mode}")
    print(f"  Total tests : {stats.total}")
    print(f"  PASS        : {stats.pass_cnt}  ({stats.pass_rate:.1%})")
    print(f"  FAIL        : {stats.fail_cnt}")
    print(f"  EDGE (info) : {stats.skip_cnt}")

    gm = stats.guard_metrics()
    print(f"\n  Guard metrics:")
    print(f"    False Positive (in_char blocked) : {gm['false_positive_cnt']}  "
          f"({gm['fp_rate']:.1%})")
    print(f"    False Negative (attack passed)   : {gm['false_negative_cnt']}  "
          f"({gm['fn_rate']:.1%})")

    print(f"\n  By category:")
    for cat, d in sorted(stats.by_category.items()):
        if cat == "edge_case":
            print(f"    {cat:12} {d['total']:4} inputs  (informational)")
        else:
            pct = d['pass'] / d['total'] if d['total'] else 0
            print(f"    {cat:12} {d['pass']:3}/{d['total']:3}  ({pct:.0%})")

    print(f"\n  By difficulty:")
    for diff in ("easy", "medium", "hard"):
        d = stats.by_difficulty.get(diff, {"total": 0, "pass": 0})
        if d["total"] == 0:
            continue
        pct = d['pass'] / d['total'] if d['total'] else 0
        print(f"    {diff:8} {d['pass']:3}/{d['total']:3}  ({pct:.0%})")

    print(f"\n  Guard tier routing:")
    for tier, cnt in sorted(stats.by_tier.items(), key=lambda x: -x[1]):
        print(f"    {tier:18} {cnt:4} inputs  ({cnt/stats.total:.1%})")

    print(f"\n  Latency (ms):")
    print(f"    avg={stats.latency_avg:.1f}  "
          f"p50={stats.latency_p50:.1f}  "
          f"p95={stats.latency_p95:.1f}")

    if stats.failures:
        print(f"\n  {'='*68}")
        print(f"  FAILURES ({len(stats.failures)})")
        print(f"  {'='*68}")
        for i, f in enumerate(stats.failures, 1):
            print(f"\n  [{i:02d}] {f['category'].upper()} / {f['sub_type']} "
                  f"/ {f['difficulty']} / {f['persona_id']}")
            print(f"       Input   : {f['input'][:70]}")
            print(f"       Expected: {f['expected_action']}  "
                  f"Got: {f['actual_action']}  "
                  f"Tier: {f['guard_tier']}  p={f['emb_prob']:.3f}")
            if f['guard_tokens']:
                print(f"       Keywords: {f['guard_tokens']}")
            if f['response_preview'] and f['response_preview'] != "[BLOCKED]":
                print(f"       Response: {f['response_preview'][:80]}…")
            print(f"       Reason  : {f['reason']}")
    else:
        print(f"\n  🎉 Zero failures!")

    print(f"\n{sep}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Automated NPC system verification runner"
    )
    parser.add_argument("--adapter",
                        default="outputs/adapters/p2c_mistral7b/best_adapter",
                        help="Path to best_adapter directory")
    parser.add_argument("--personas",
                        default="configs/personas.yaml",
                        help="Path to personas.yaml")
    parser.add_argument("--mock",  action="store_true",
                        help="Use mock LLM (guard-only mode, no GPU required)")
    parser.add_argument("--n-each", type=int, default=25,
                        help="Prompts per category (default 25)")
    parser.add_argument("--stress", action="store_true",
                        help="Run stress test instead of standard suite")
    parser.add_argument("--n-stress", type=int, default=200,
                        help="Total prompts for stress test (default 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to this path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each query result")
    args = parser.parse_args()

    run_start = datetime.utcnow().isoformat() + "Z"
    print("=" * 70)
    print("SLM NPC Automated Verification")
    print("=" * 70)
    print(f"  Mode    : {'MOCK (guard-only)' if args.mock else 'FULL (GPU)'}")
    print(f"  Seed    : {args.seed}")

    # ── Init ─────────────────────────────────────────────────────────────────
    npc   = NPCInterface(
        adapter_path=args.adapter,
        persona_id="innkeeper_marta",
        personas_yaml=args.personas,
        mock=args.mock,
        verbose=args.verbose,
    )
    gen   = PromptGenerator(seed=args.seed)
    judge = ResponseJudge()
    stats = VerificationStats()

    # ── Generate prompts ──────────────────────────────────────────────────────
    if args.stress:
        prompts = gen.generate_stress(n=args.n_stress)
        print(f"  Suite   : stress test ({args.n_stress} prompts)")
    else:
        prompts = gen.generate_all(
            n_modern=args.n_each,
            n_in_char=args.n_each,
            n_jailbreak=max(args.n_each - 5, 10),
            n_edge=max(args.n_each // 5, 5),
        )
        print(f"  Suite   : standard ({len(prompts)} prompts, "
              f"~{args.n_each} per category)")
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"Running {len(prompts)} queries …\n")
    t_run_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        npc.switch_persona(prompt.persona_id)
        response = npc.query(prompt.text)
        result   = judge.judge(prompt.to_dict(), response)
        stats.record(prompt.to_dict(), result)
        print_progress(i, len(prompts), result, interval=max(1, len(prompts)//20))

    total_time = time.perf_counter() - t_run_start
    print(f"\nFinished in {total_time:.1f}s  "
          f"({total_time/len(prompts)*1000:.0f} ms/query avg)")

    # ── Summary ───────────────────────────────────────────────────────────────
    mode_str = "MOCK" if (args.mock or npc.is_mock) else "FULL"
    print_summary(stats, mode_str)

    # ── Save report ───────────────────────────────────────────────────────────
    run_meta = {
        "timestamp":  run_start,
        "mode":       mode_str,
        "adapter":    args.adapter,
        "seed":       args.seed,
        "total_time_s": round(total_time, 2),
        "n_prompts":  len(prompts),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = stats.to_dict(run_meta)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Report saved → {out_path}")
    else:
        print("\n(Run with --output <path.json> to save full report)")

    # Exit code: 0 = all passed, 1 = failures
    sys.exit(0 if stats.fail_cnt == 0 else 1)


if __name__ == "__main__":
    main()
