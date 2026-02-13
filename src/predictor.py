from __future__ import annotations

from dataclasses import dataclass
import argparse
import fnmatch
import json
from pathlib import Path
import re
import subprocess
from typing import List


BASELINE_PATTERNS = {
    "systems": "Systems-oriented software engineer; prefers low-level understanding",
    "performance": "Optimizes for performance, efficiency, and architectural correctness",
    "design_then_iterate": "Designs first, then iterates quickly during implementation",
    "minimal_frameworks": "Avoids unnecessary frameworks and over-engineering",
    "long_horizon": "Works on complex, long-horizon, research-like projects",
    "task_switch_when_blocked": "Switches tasks when blocked, not due to novelty-seeking",
    "pragmatic_elegance": "Prefers pragmatic but technically elegant solutions",
}
DEFAULT_CONTEXT_PATTERNS = ["*.log", "*.txt", "*.md"]
DEFAULT_IGNORED_DIRS = [
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
]
DEFAULT_MAX_FILE_BYTES = 1_000_000
SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}
SKIP_REASON_SEVERITY = {
    "read_error": "high",
    "stat_error": "high",
    "missing_or_not_directory": "high",
    "missing_or_not_file": "medium",
    "over_max_bytes": "medium",
    "over_max_files": "medium",
    "empty_file": "low",
    "ignored_dir": "low",
    "ignored_pattern": "low",
    "duplicate_candidate": "low",
    "not_a_file": "low",
}


@dataclass
class RankedItem:
    title: str
    probability: float
    rationale: str


@dataclass
class PredictionReport:
    likely_next_actions: List[RankedItem]
    current_intent_inference: str
    decision_predictions: List[str]
    style_deviations_detected: List[str]
    reasoning_trace: List[str]
    alternative_paths: List[RankedItem]

    def to_markdown(self) -> str:
        lines = []
        lines.append("## Likely Next Actions (ranked)")
        for idx, item in enumerate(self.likely_next_actions, start=1):
            lines.append(
                f"{idx}. {item.title} - {int(item.probability * 100)}% likely. {item.rationale}"
            )

        lines.append("\n## Current Intent Inference")
        lines.append(self.current_intent_inference)

        lines.append("\n## Decision Predictions")
        for idx, decision in enumerate(self.decision_predictions, start=1):
            lines.append(f"{idx}. {decision}")

        lines.append("\n## Style Deviations Detected")
        if self.style_deviations_detected:
            for idx, dev in enumerate(self.style_deviations_detected, start=1):
                lines.append(f"{idx}. {dev}")
        else:
            lines.append("1. No strong deviations detected from baseline engineering style.")

        lines.append("\n## Reasoning Trace")
        for idx, reason in enumerate(self.reasoning_trace, start=1):
            lines.append(f"{idx}. {reason}")

        lines.append("\n## Alternative Paths")
        for idx, alt in enumerate(self.alternative_paths, start=1):
            lines.append(
                f"{idx}. {alt.title} - {int(alt.probability * 100)}% likely. Trigger: {alt.rationale}"
            )

        return "\n".join(lines)


class DigitalTwinPredictor:
    """Deterministic, evidence-based predictor for technical work patterns."""

    def __init__(self, baseline: dict | None = None):
        self.baseline = baseline or BASELINE_PATTERNS

    def predict(self, context: str) -> PredictionReport:
        ctx = context.strip()
        lctx = ctx.lower()

        action_scores = {
            "Write or update a concise architecture/design note before large edits": 0.58,
            "Run focused profiling/benchmarking on current bottlenecks": 0.54,
            "Implement the smallest viable change that unlocks blocked flow": 0.50,
            "Add targeted regression/performance tests around changed behavior": 0.48,
            "Refactor hot-path code to reduce abstraction overhead": 0.46,
            "Temporarily switch to a parallel subtask while preserving context": 0.42,
        }

        reasoning = []
        deviations = []
        decisions = []

        # Signal extraction from context text
        has_benchmark = bool(re.search(r"benchmark|latency|throughput|perf", lctx))
        has_bug = bool(re.search(r"bug|regression|failure|error|panic|exception", lctx))
        has_arch = bool(re.search(r"architecture|design|diagram|adr", lctx))
        has_blocked = bool(re.search(r"blocked|stuck|waiting|dependency", lctx))
        has_framework = bool(re.search(r"framework|boilerplate|scaffold|library", lctx))
        has_long_horizon = bool(re.search(r"roadmap|research|long-term|phd|prototype", lctx))
        has_logs = bool(re.search(r"log|trace|stack", lctx))
        has_db = bool(re.search(r"db|database|query|index|storage", lctx))

        if has_benchmark:
            action_scores["Run focused profiling/benchmarking on current bottlenecks"] += 0.22
            action_scores["Refactor hot-path code to reduce abstraction overhead"] += 0.10
            decisions.append("Prefer measuring before rewriting; profile first, then optimize the top hotspot only.")
            reasoning.append("Performance terms in context indicate optimization-first sequencing.")

        if has_bug:
            action_scores["Add targeted regression/performance tests around changed behavior"] += 0.20
            action_scores["Implement the smallest viable change that unlocks blocked flow"] += 0.08
            decisions.append("Constrain blast radius with minimal, test-backed fixes before broader refactors.")
            reasoning.append("Failure signals imply immediate containment and reproducibility work.")

        if has_arch:
            action_scores["Write or update a concise architecture/design note before large edits"] += 0.24
            decisions.append("Keep architecture artifacts current to preserve long-term correctness under iteration.")
            reasoning.append("Architecture references align with design-first behavior.")

        if has_blocked:
            action_scores["Temporarily switch to a parallel subtask while preserving context"] += 0.24
            decisions.append("Pivot to an unblocked subsystem while capturing exact blocker state for fast return.")
            reasoning.append("Blocker language maps to strategic task switching rather than idle context switching.")

        if has_framework:
            deviations.append(
                "Context includes framework-heavy language; baseline style usually favors lighter-weight, purpose-fit abstractions."
            )
            reasoning.append("Potential mismatch detected: framework reliance vs minimal-framework preference.")

        if has_long_horizon:
            action_scores["Write or update a concise architecture/design note before large edits"] += 0.06
            decisions.append("Bias toward modular boundaries and explicit interfaces for long-horizon maintainability.")
            reasoning.append("Long-horizon signals increase emphasis on architecture and interface stability.")

        if has_logs:
            decisions.append("Use low-level traces/log correlation before changing core logic to avoid blind edits.")
            reasoning.append("Trace/log evidence suggests root-cause-first debugging path.")

        if has_db:
            decisions.append("Evaluate query/index tradeoffs before scaling app-layer complexity.")
            reasoning.append("Storage/query terms imply system-level bottleneck analysis.")

        if not reasoning:
            reasoning.append(
                "No high-specificity technical signals found; defaulting to baseline sequence: design note -> targeted implementation -> validation."
            )

        ranked = self._rank(action_scores)

        intent = self._infer_intent(has_benchmark, has_bug, has_arch, has_db, has_long_horizon)

        if not decisions:
            decisions = [
                "Decompose work into measurable increments with explicit success criteria.",
                "Pick low-overhead primitives over heavy frameworks unless complexity justifies them.",
                "Protect hot paths with microbenchmarks and regression tests before deep optimization.",
            ]

        alternatives = self._alternatives(has_benchmark, has_bug, has_blocked, has_arch)

        return PredictionReport(
            likely_next_actions=ranked,
            current_intent_inference=intent,
            decision_predictions=decisions,
            style_deviations_detected=deviations,
            reasoning_trace=reasoning,
            alternative_paths=alternatives,
        )

    @staticmethod
    def _rank(action_scores: dict) -> List[RankedItem]:
        top = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)[:6]
        max_score = max(score for _, score in top)
        min_score = min(score for _, score in top)
        span = max(max_score - min_score, 1e-6)

        ranked = []
        for title, score in top:
            norm = 0.55 + ((score - min_score) / span) * 0.35
            ranked.append(
                RankedItem(
                    title=title,
                    probability=round(min(max(norm, 0.01), 0.99), 2),
                    rationale="Score driven by detected context signals and baseline engineering tendencies.",
                )
            )
        return ranked

    @staticmethod
    def _infer_intent(
        has_benchmark: bool,
        has_bug: bool,
        has_arch: bool,
        has_db: bool,
        has_long_horizon: bool,
    ) -> str:
        if has_bug and has_benchmark:
            return (
                "High probability objective: stabilize correctness first, then recover performance by removing bottlenecks with measured changes."
            )
        if has_arch and has_long_horizon:
            return (
                "High probability objective: shape a durable architecture for a long-term system while iterating on near-term deliverables."
            )
        if has_db and has_benchmark:
            return (
                "High probability objective: optimize end-to-end system throughput, with storage/query behavior as a probable bottleneck."
            )
        if has_bug:
            return (
                "High probability objective: isolate and fix a regression quickly with minimal surface-area changes."
            )
        return (
            "Most likely objective: progress a technically complex feature via design-first planning and pragmatic implementation increments."
        )

    @staticmethod
    def _alternatives(
        has_benchmark: bool,
        has_bug: bool,
        has_blocked: bool,
        has_arch: bool,
    ) -> List[RankedItem]:
        options = [
            RankedItem(
                title="Debug-first branch",
                probability=0.70 if has_bug else 0.38,
                rationale="Critical failures or regressions become reproducible test cases and immediate fixes.",
            ),
            RankedItem(
                title="Performance-first branch",
                probability=0.72 if has_benchmark else 0.42,
                rationale="Observed latency/throughput issues trigger profiling and hot-path refactors.",
            ),
            RankedItem(
                title="Architecture-first branch",
                probability=0.68 if has_arch else 0.45,
                rationale="Ambiguous boundaries or scale concerns trigger interface and module redesign.",
            ),
            RankedItem(
                title="Parallel-subtask branch",
                probability=0.66 if has_blocked else 0.36,
                rationale="External blockers trigger context-preserving task switch to maintain momentum.",
            ),
        ]
        return sorted(options, key=lambda x: x.probability, reverse=True)[:3]


def _skip(path: str, reason: str) -> dict:
    severity = SKIP_REASON_SEVERITY.get(reason, "low")
    return {
        "path": path,
        "reason": reason,
        "severity": severity,
        "severity_score": SEVERITY_RANK[severity],
    }


def summarize_skips(skipped_files: List[dict]) -> dict:
    by_reason = {}
    by_severity = {"high": 0, "medium": 0, "low": 0}
    for item in skipped_files:
        reason = item.get("reason", "unknown")
        severity = item.get("severity", SKIP_REASON_SEVERITY.get(reason, "low"))
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
    return {"by_reason": by_reason, "by_severity": by_severity}


def sort_skips_by_severity(skipped_files: List[dict]) -> List[dict]:
    return sorted(
        skipped_files,
        key=lambda item: (
            -item.get(
                "severity_score",
                SEVERITY_RANK.get(
                    item.get("severity", SKIP_REASON_SEVERITY.get(item.get("reason", ""), "low")),
                    1,
                ),
            ),
            item.get("reason", ""),
            item.get("path", ""),
        ),
    )


def gather_file_context_with_metadata(paths: List[str]) -> tuple[str, List[str], List[dict]]:
    chunks = []
    included_files = []
    skipped_files = []
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path)
        if not path.exists() or not path.is_file():
            skipped_files.append(_skip(key, "missing_or_not_file"))
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            skipped_files.append(_skip(key, "read_error"))
            continue
        if not content:
            skipped_files.append(_skip(key, "empty_file"))
            continue
        chunks.append(f"[file:{path}]\n{content}")
        included_files.append(key)

    return "\n\n".join(chunks), included_files, skipped_files


def gather_file_context(paths: List[str]) -> str:
    text, _, _ = gather_file_context_with_metadata(paths)
    return text


def discover_context_files_with_metadata(
    from_dir: str,
    patterns: List[str] | None = None,
    max_files: int = 10,
    recursive: bool = True,
    ignore_patterns: List[str] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ignore_dirs: List[str] | None = None,
) -> tuple[List[str], List[dict]]:
    dir_path = Path(from_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        return [], [_skip(str(dir_path), "missing_or_not_directory")]

    ignore_patterns = ignore_patterns or []
    ignore_dirs_set = set(ignore_dirs or [])
    skipped_files = []

    selected = {}
    seen_paths = set()
    for pattern in (patterns or DEFAULT_CONTEXT_PATTERNS):
        iterator = dir_path.rglob(pattern) if recursive else dir_path.glob(pattern)
        for path in iterator:
            key = str(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            if not path.is_file():
                skipped_files.append(_skip(key, "not_a_file"))
                continue
            try:
                rel = path.relative_to(dir_path).as_posix()
                rel_parts = path.relative_to(dir_path).parts
            except ValueError:
                rel = path.as_posix()
                rel_parts = path.parts
            if any(part in ignore_dirs_set for part in rel_parts):
                skipped_files.append(_skip(key, "ignored_dir"))
                continue
            if ignore_patterns and (
                any(fnmatch.fnmatch(path.name, p) for p in ignore_patterns)
                or any(fnmatch.fnmatch(rel, p) for p in ignore_patterns)
            ):
                skipped_files.append(_skip(key, "ignored_pattern"))
                continue
            try:
                stat = path.stat()
            except OSError:
                skipped_files.append(_skip(key, "stat_error"))
                continue
            if stat.st_size > max(max_file_bytes, 1):
                skipped_files.append(_skip(key, "over_max_bytes"))
                continue
            mtime = stat.st_mtime
            if key not in selected:
                selected[key] = mtime

    if not selected:
        return [], skipped_files

    ranked = sorted(selected.items(), key=lambda item: item[1], reverse=True)
    limit = max(max_files, 1)
    included = [path for path, _ in ranked[:limit]]
    skipped_from_limit = [_skip(path, "over_max_files") for path, _ in ranked[limit:]]
    return included, skipped_files + skipped_from_limit


def discover_context_files(
    from_dir: str,
    patterns: List[str] | None = None,
    max_files: int = 10,
    recursive: bool = True,
    ignore_patterns: List[str] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ignore_dirs: List[str] | None = None,
) -> List[str]:
    files, _ = discover_context_files_with_metadata(
        from_dir=from_dir,
        patterns=patterns,
        max_files=max_files,
        recursive=recursive,
        ignore_patterns=ignore_patterns,
        max_file_bytes=max_file_bytes,
        ignore_dirs=ignore_dirs,
    )
    return files


def gather_git_context(max_commits: int = 8) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"--max-count={max_commits}",
                "--pretty=format:%h %ad %s",
                "--date=short",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""

    if result.returncode != 0:
        return ""

    output = result.stdout.strip()
    if not output:
        return ""

    return f"[git-log:last-{max_commits}-commits]\n{output}"


def resolve_context_with_metadata(
    inline_context: str,
    from_files: List[str] | None = None,
    from_dir: str | None = None,
    dir_patterns: List[str] | None = None,
    dir_max_files: int = 10,
    dir_recursive: bool = True,
    dir_ignore_patterns: List[str] | None = None,
    dir_max_bytes: int = DEFAULT_MAX_FILE_BYTES,
    dir_ignore_dirs: List[str] | None = None,
    from_git: bool = False,
    git_commits: int = 8,
) -> tuple[str, dict]:
    parts = []
    scan = {
        "included_files": [],
        "skipped_files": [],
        "skip_summary": {"by_reason": {}, "by_severity": {"high": 0, "medium": 0, "low": 0}},
        "from_git": {"requested": from_git, "included": False, "reason": ""},
    }

    clean_inline = inline_context.strip()
    if clean_inline:
        parts.append(clean_inline)

    candidate_files = []
    dedupe_seen = set()
    if from_files:
        for raw in from_files:
            key = str(Path(raw))
            if key in dedupe_seen:
                scan["skipped_files"].append(_skip(key, "duplicate_candidate"))
                continue
            dedupe_seen.add(key)
            candidate_files.append(key)
    if from_dir:
        discovered_files, discover_skips = discover_context_files_with_metadata(
            from_dir=from_dir,
            patterns=dir_patterns,
            max_files=dir_max_files,
            recursive=dir_recursive,
            ignore_patterns=dir_ignore_patterns,
            max_file_bytes=dir_max_bytes,
            ignore_dirs=dir_ignore_dirs,
        )
        scan["skipped_files"].extend(discover_skips)
        for raw in discovered_files:
            key = str(Path(raw))
            if key in dedupe_seen:
                scan["skipped_files"].append(_skip(key, "duplicate_candidate"))
                continue
            dedupe_seen.add(key)
            candidate_files.append(key)

    if candidate_files:
        file_ctx, included_files, read_skips = gather_file_context_with_metadata(
            candidate_files
        )
        scan["included_files"].extend(included_files)
        scan["skipped_files"].extend(read_skips)
        if file_ctx:
            parts.append(file_ctx)

    if from_git:
        git_ctx = gather_git_context(max_commits=git_commits)
        if git_ctx:
            scan["from_git"]["included"] = True
            parts.append(git_ctx)
        else:
            scan["from_git"]["reason"] = "git_unavailable_or_no_history"

    scan["skipped_files"] = sort_skips_by_severity(scan["skipped_files"])
    scan["skip_summary"] = summarize_skips(scan["skipped_files"])

    return "\n\n".join(parts).strip(), scan


def resolve_context(
    inline_context: str,
    from_files: List[str] | None = None,
    from_dir: str | None = None,
    dir_patterns: List[str] | None = None,
    dir_max_files: int = 10,
    dir_recursive: bool = True,
    dir_ignore_patterns: List[str] | None = None,
    dir_max_bytes: int = DEFAULT_MAX_FILE_BYTES,
    dir_ignore_dirs: List[str] | None = None,
    from_git: bool = False,
    git_commits: int = 8,
) -> str:
    context, _ = resolve_context_with_metadata(
        inline_context=inline_context,
        from_files=from_files,
        from_dir=from_dir,
        dir_patterns=dir_patterns,
        dir_max_files=dir_max_files,
        dir_recursive=dir_recursive,
        dir_ignore_patterns=dir_ignore_patterns,
        dir_max_bytes=dir_max_bytes,
        dir_ignore_dirs=dir_ignore_dirs,
        from_git=from_git,
        git_commits=git_commits,
    )
    return context


def format_scan_metadata(scan: dict) -> str:
    lines = []
    skip_summary = scan.get("skip_summary", summarize_skips(scan["skipped_files"]))
    skipped_files = sort_skips_by_severity(scan["skipped_files"])

    lines.append("## Scan Metadata")
    lines.append(f"1. Included files: {len(scan['included_files'])}")
    lines.append(f"2. Skipped files: {len(skipped_files)}")
    lines.append(
        "3. Skip severities: "
        f"high={skip_summary['by_severity'].get('high', 0)}, "
        f"medium={skip_summary['by_severity'].get('medium', 0)}, "
        f"low={skip_summary['by_severity'].get('low', 0)}"
    )

    git_state = scan["from_git"]
    git_msg = (
        "included"
        if git_state["included"]
        else (
            f"not included ({git_state['reason']})"
            if git_state["requested"]
            else "not requested"
        )
    )
    lines.append(f"4. Git context: {git_msg}")

    if skip_summary["by_reason"]:
        lines.append("\nTop Skip Reasons:")
        top_reasons = sorted(
            skip_summary["by_reason"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        for reason, count in top_reasons:
            sev = SKIP_REASON_SEVERITY.get(reason, "low")
            lines.append(f"- {reason}: {count} ({sev})")

    if scan["included_files"]:
        lines.append("\nIncluded File Paths:")
        for path in scan["included_files"]:
            lines.append(f"- {path}")

    if skipped_files:
        lines.append("\nSkipped File Paths:")
        for item in skipped_files:
            lines.append(
                f"- {item['path']} [{item['reason']}] severity={item['severity']}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral Digital Twin / Future Work Predictor"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Current project context text. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown.",
    )
    parser.add_argument(
        "--from-files",
        nargs="*",
        default=[],
        help="Read additional context from files (logs, notes, issue text).",
    )
    parser.add_argument(
        "--from-dir",
        type=str,
        default="",
        help="Auto-discover context files from this directory.",
    )
    parser.add_argument(
        "--dir-patterns",
        nargs="*",
        default=DEFAULT_CONTEXT_PATTERNS,
        help="File patterns for --from-dir (default: *.log *.txt *.md).",
    )
    parser.add_argument(
        "--dir-max-files",
        type=int,
        default=10,
        help="Max number of recent files loaded from --from-dir.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top level of --from-dir (no recursive scan).",
    )
    parser.add_argument(
        "--dir-max-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Ignore files larger than this size when scanning --from-dir.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=[],
        help="Optional glob patterns to exclude when scanning --from-dir.",
    )
    parser.add_argument(
        "--ignore-dirs",
        nargs="*",
        default=DEFAULT_IGNORED_DIRS,
        help="Directory names to skip during --from-dir scanning.",
    )
    parser.add_argument(
        "--from-git",
        action="store_true",
        help="Include recent git commit messages as context.",
    )
    parser.add_argument(
        "--git-commits",
        type=int,
        default=8,
        help="How many recent commits to ingest when using --from-git.",
    )
    parser.add_argument(
        "--show-scan",
        action="store_true",
        help="Show metadata for ingested and skipped files.",
    )

    args = parser.parse_args()

    context, scan = resolve_context_with_metadata(
        inline_context=args.context,
        from_files=args.from_files,
        from_dir=args.from_dir.strip() or None,
        dir_patterns=args.dir_patterns,
        dir_max_files=max(args.dir_max_files, 1),
        dir_recursive=not args.non_recursive,
        dir_ignore_patterns=args.ignore_patterns,
        dir_max_bytes=max(args.dir_max_bytes, 1),
        dir_ignore_dirs=args.ignore_dirs,
        from_git=args.from_git,
        git_commits=max(args.git_commits, 1),
    )
    if not context:
        context = input("Paste context: ").strip()

    predictor = DigitalTwinPredictor()
    report = predictor.predict(context)

    if args.json:
        payload = {
            "likely_next_actions": [item.__dict__ for item in report.likely_next_actions],
            "current_intent_inference": report.current_intent_inference,
            "decision_predictions": report.decision_predictions,
            "style_deviations_detected": report.style_deviations_detected,
            "reasoning_trace": report.reasoning_trace,
            "alternative_paths": [item.__dict__ for item in report.alternative_paths],
        }
        if args.show_scan:
            payload["scan"] = scan
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
    else:
        print(report.to_markdown())
        if args.show_scan:
            print()
            print(format_scan_metadata(scan))


if __name__ == "__main__":
    main()
