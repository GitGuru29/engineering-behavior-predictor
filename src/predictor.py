from __future__ import annotations

from dataclasses import dataclass
import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
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
    "outside_from_dir": "low",
    "outside_dir_patterns": "low",
}
SKIP_REASON_HINTS = {
    "read_error": "Verify file permissions and file encoding before rerunning.",
    "stat_error": "Check filesystem access and retry with accessible paths.",
    "missing_or_not_directory": "Fix --from-dir path or create the target directory.",
    "missing_or_not_file": "Fix file path typos or regenerate missing artifact files.",
    "over_max_bytes": "Increase --dir-max-bytes or narrow file patterns.",
    "over_max_files": "Increase --dir-max-files or tighten --dir-patterns.",
    "empty_file": "Regenerate logs/notes so files contain meaningful context.",
    "ignored_dir": "Remove directory from --ignore-dirs if it must be scanned.",
    "ignored_pattern": "Adjust --ignore-patterns to include this file.",
    "duplicate_candidate": "Remove duplicate file inputs from --from-files.",
    "not_a_file": "Provide only regular files, not directories or device paths.",
    "outside_from_dir": "Adjust --from-dir to cover these files if needed.",
    "outside_dir_patterns": "Widen --dir-patterns if these files should be considered.",
}
DEFAULT_ACTION_SCORES = {
    "Write or update a concise architecture/design note before large edits": 0.58,
    "Run focused profiling/benchmarking on current bottlenecks": 0.54,
    "Implement the smallest viable change that unlocks blocked flow": 0.50,
    "Add targeted regression/performance tests around changed behavior": 0.48,
    "Refactor hot-path code to reduce abstraction overhead": 0.46,
    "Temporarily switch to a parallel subtask while preserving context": 0.42,
}
DEFAULT_SIGNAL_PATTERNS = {
    "benchmark": r"benchmark|latency|throughput|perf",
    "bug": r"bug|regression|failure|error|panic|exception",
    "arch": r"architecture|design|diagram|adr",
    "blocked": r"blocked|stuck|waiting|dependency",
    "framework": r"\bframework\b|boilerplate|over-?engineer|heavy stack|abstraction layer",
    "long_horizon": r"roadmap|research|long-term|phd|prototype",
    "logs": r"log|trace|stack",
    "db": r"db|database|query|index|storage",
}
DEFAULT_SIGNAL_BOOSTS = {
    "benchmark": {
        "Run focused profiling/benchmarking on current bottlenecks": 0.22,
        "Refactor hot-path code to reduce abstraction overhead": 0.10,
    },
    "bug": {
        "Add targeted regression/performance tests around changed behavior": 0.20,
        "Implement the smallest viable change that unlocks blocked flow": 0.08,
    },
    "arch": {
        "Write or update a concise architecture/design note before large edits": 0.24,
    },
    "blocked": {
        "Temporarily switch to a parallel subtask while preserving context": 0.24,
    },
    "long_horizon": {
        "Write or update a concise architecture/design note before large edits": 0.06,
    },
}
SOURCE_CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".kt",
    ".kts",
    ".m",
    ".mm",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
}
DOC_EXTENSIONS = {".adoc", ".md", ".rst"}
TEXT_EXTENSIONS = {".txt"}
LOG_EXTENSIONS = {".log", ".trace"}
CONFIG_EXTENSIONS = {".cfg", ".conf", ".ini", ".json", ".toml", ".xml", ".yaml", ".yml"}
DEFAULT_CONTEXT_SOURCE_WEIGHTS = {
    "inline": 1.0,
    "git-log": 1.0,
    "log": 1.0,
    "text": 0.95,
    "doc": 0.45,
    "config": 0.35,
    "source": 0.08,
    "test": 0.05,
    "other_file": 0.40,
}
DEFAULT_SIGNAL_THRESHOLDS = {
    "benchmark": 0.85,
    "bug": 0.90,
    "arch": 0.95,
    "blocked": 0.90,
    "framework": 0.85,
    "long_horizon": 0.95,
    "logs": 0.85,
    "db": 0.85,
}
EXTRA_SIGNAL_PATTERNS = {
    "android": (
        r"androidmanifest|build\.gradle|gradle\.kts|mainactivity|viewmodel|compose|jetpack|kotlin|"
        r"\.kt\b|activity|fragment|room|retrofit"
    ),
    "ui_flows": r"screen|navigation|checkout|cart|auth|signin|signup",
    "framework_overreach": r"boilerplate|scaffold|over-?engineer|heavy stack|abstraction layer",
    "delivery_markers": r"\bfeat|fix|refactor|perf|test|docs|chore\b",
}
DEFAULT_EXTRA_SIGNAL_THRESHOLDS = {
    "android": 0.85,
    "ui_flows": 0.90,
    "framework_overreach": 0.85,
    "delivery_markers": 0.90,
}
ANDROID_ACTION_PRIORS = {
    "Add instrumentation/UI tests for critical user journeys (auth, cart, checkout)": 0.62,
    "Profile startup and scroll performance on representative Android devices": 0.58,
    "Harden ViewModel state and side-effect handling for Compose screens": 0.56,
}


@dataclass
class RankedItem:
    title: str
    probability: float
    rationale: str


@dataclass
class ContextSection:
    source_type: str
    source_ref: str
    weight: float
    text: str


@dataclass
class PredictionReport:
    likely_next_actions: List[RankedItem]
    current_intent_inference: str
    decision_predictions: List[str]
    future_improvements: List[RankedItem]
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

        lines.append("\n## Future Improvements (ranked)")
        for idx, item in enumerate(self.future_improvements, start=1):
            lines.append(
                f"{idx}. {item.title} - {int(item.probability * 100)}% likely value. {item.rationale}"
            )

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


@dataclass
class FounderAuditSection:
    title: str
    recommendations: List[str]


@dataclass
class FounderAuditReport:
    sections: List[FounderAuditSection]
    analyzed_files: int
    index_truncated: bool

    def to_dict(self) -> dict:
        return {
            "sections": [
                {"title": section.title, "recommendations": section.recommendations}
                for section in self.sections
            ],
            "analyzed_files": self.analyzed_files,
            "index_truncated": self.index_truncated,
        }

    def to_markdown(self) -> str:
        lines = ["## Project Improvement Audit"]
        for idx, section in enumerate(self.sections, start=1):
            lines.append(f"{idx}. {section.title}")
            for rec_idx, recommendation in enumerate(section.recommendations, start=1):
                lines.append(f"{rec_idx}. {recommendation}")
        lines.append(f"\nAudit coverage: {self.analyzed_files} files indexed.")
        if self.index_truncated:
            lines.append("Index cap reached; raise --audit-max-files for broader coverage.")
        return "\n".join(lines)


def _collect_repo_index(
    project_dir: str,
    ignore_dirs: List[str],
    max_files: int,
) -> tuple[List[str], bool]:
    files = []
    truncated = False
    base = Path(project_dir).resolve()
    ignore_dirs_set = set(ignore_dirs)

    for root, dirs, filenames in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ignore_dirs_set]
        root_path = Path(root)
        for filename in filenames:
            rel = (root_path / filename).relative_to(base).as_posix()
            files.append(rel)
            if len(files) >= max(max_files, 1):
                truncated = True
                return files, truncated
    return files, truncated


def _read_text_if_small(path: Path, max_bytes: int = 300_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _find_repo_file(index: List[str], target_name: str) -> str:
    target_lower = target_name.lower()
    for rel in index:
        if rel.lower() == target_lower:
            return rel
    return ""


def _extract_android_permissions(manifest_text: str) -> List[str]:
    return re.findall(
        r"uses-permission[^>]*android:name=[\"']([^\"']+)[\"']",
        manifest_text,
        flags=re.IGNORECASE,
    )


def _extract_gradle_modules(settings_text: str) -> List[str]:
    matches = re.findall(r"[\"'](:[^\"']+)[\"']", settings_text)
    modules = []
    seen = set()
    for item in matches:
        if item in seen:
            continue
        seen.add(item)
        modules.append(item)
    return modules


def build_founder_audit_report(
    project_dir: str,
    context: str,
    scan: dict,
    ignore_dirs: List[str] | None = None,
    max_files: int = 5000,
) -> FounderAuditReport:
    repo_index, truncated = _collect_repo_index(
        project_dir=project_dir,
        ignore_dirs=ignore_dirs or DEFAULT_IGNORED_DIRS,
        max_files=max_files,
    )
    index_lower = [path.lower() for path in repo_index]
    context_lower = context.lower()

    readme_rel = _find_repo_file(repo_index, "README.md")
    readme_text = (
        _read_text_if_small(Path(project_dir) / readme_rel) if readme_rel else ""
    )
    readme_lower = readme_text.lower()

    settings_rel = _find_repo_file(repo_index, "settings.gradle.kts") or _find_repo_file(
        repo_index, "settings.gradle"
    )
    settings_text = (
        _read_text_if_small(Path(project_dir) / settings_rel) if settings_rel else ""
    )
    modules = _extract_gradle_modules(settings_text)

    manifest_rel = ""
    for rel in repo_index:
        if rel.lower().endswith("androidmanifest.xml"):
            manifest_rel = rel
            break
    manifest_text = (
        _read_text_if_small(Path(project_dir) / manifest_rel) if manifest_rel else ""
    )
    permissions = _extract_android_permissions(manifest_text)

    has_gradle_files = any(
        path.endswith("build.gradle") or path.endswith("build.gradle.kts")
        for path in index_lower
    )
    has_kotlin_or_java = any(path.endswith(".kt") or path.endswith(".java") for path in index_lower)
    has_android = bool(manifest_rel or (has_gradle_files and has_kotlin_or_java))
    has_readme = bool(readme_rel)
    has_problem = bool(re.search(r"\bproblem\b|pain point|challenge", readme_lower))
    has_solution = bool(re.search(r"\bsolution\b|approach", readme_lower))
    has_features = bool(re.search(r"\bfeatures?\b", readme_lower))
    has_demo = bool(re.search(r"\bdemo\b|screenshot|video|gif", readme_lower))
    has_target_user = bool(
        re.search(r"target user|audience|who (is|this) for|for developers|for teams", readme_lower)
    )
    has_differentiator = bool(
        re.search(r"\bwhy\b|differentiator|unique|compared to|vs\.", readme_lower)
    )
    has_roadmap = bool(re.search(r"roadmap|future work|next steps", readme_lower))
    has_run = bool(re.search(r"getting started|setup|install|run|quick start", readme_lower))
    has_env_example = any(
        Path(rel).name.lower() in {".env.example", ".env.sample", ".env.template"}
        for rel in repo_index
    )
    has_runtime_config = any(
        Path(rel).name.lower()
        in {
            ".env",
            "local.properties",
            "gradle.properties",
            "application.properties",
            "secrets.properties",
        }
        for rel in repo_index
    )
    has_gradlew = any(Path(rel).name == "gradlew" for rel in repo_index)
    has_ci = any(
        rel.startswith(".github/workflows/")
        or rel == ".gitlab-ci.yml"
        or rel.lower() == "jenkinsfile"
        for rel in repo_index
    )
    has_tests = any(
        path.startswith("test/")
        or path.startswith("tests/")
        or "/test/" in path
        or "/tests/" in path
        or path.endswith("test.kt")
        or path.endswith("test.java")
        or path.endswith("_test.py")
        for path in index_lower
    )
    has_ui_tests = any(
        "/androidtest/" in path
        or ("ui" in path and "test" in path and (path.endswith(".kt") or path.endswith(".java")))
        for path in index_lower
    )
    has_lint = any(
        Path(rel).name.lower() in {".editorconfig", "detekt.yml", "detekt.yaml", ".eslintrc", "pyproject.toml"}
        or "ktlint" in rel.lower()
        for rel in repo_index
    )
    has_benchmark = any("benchmark" in path for path in index_lower)
    has_baseline_profile = any(
        "baselineprofile" in path or "baseline-profile" in path for path in index_lower
    )
    has_cache_markers = bool(re.search(r"\bcache|memo|lru|ttl\b", context_lower))
    has_db_markers = bool(re.search(r"\broom|sqlite|dao|database|query\b", context_lower))
    has_network_markers = bool(re.search(r"\bretrofit|okhttp|http|api\b", context_lower))

    layer_signals = {
        "ui": any("/ui/" in path or "screen" in path for path in index_lower),
        "domain": any(
            "/domain/" in path or "usecase" in path or "interactor" in path for path in index_lower
        ),
        "data": any("/data/" in path or "/repository/" in path or "datasource" in path for path in index_lower),
        "infra": any("/infra/" in path or "/network/" in path or "/di/" in path for path in index_lower),
    }
    utils_files = [
        rel
        for rel in repo_index
        if "/utils/" in rel.lower() or "/util/" in rel.lower()
    ]

    large_code_files = []
    for abs_path in scan.get("included_files", [])[:120]:
        p = Path(abs_path)
        if not p.exists() or not p.is_file():
            continue
        try:
            line_count = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        if line_count >= 450:
            large_code_files.append((str(p), line_count))
    large_code_files.sort(key=lambda item: item[1], reverse=True)

    secret_hits = re.findall(
        r"(?im)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"'][^\"']{8,}[\"']",
        context,
    )

    first_impression = []
    if not has_readme:
        first_impression.append(
            "No README.md found. Add a top section with one-line value proposition, problem, solution, and three core features."
        )
    else:
        missing_top = []
        if not has_problem:
            missing_top.append("problem")
        if not has_solution:
            missing_top.append("solution")
        if not has_features:
            missing_top.append("features")
        if not has_demo:
            missing_top.append("demo")
        if missing_top:
            first_impression.append(
                "README exists but top-of-file pitch is incomplete (missing: "
                + ", ".join(missing_top)
                + "). Rewrite first 25 lines as Problem -> Solution -> Features -> Demo."
            )
    if not has_demo:
        first_impression.append(
            "No demo evidence detected. Add 2-4 screenshots or a 30-60s walkthrough video link in README."
        )
    if not first_impression:
        first_impression.append(
            "README first impression is already strong; keep repo title/tagline and demo links synchronized with current feature scope."
        )

    project_structure = []
    if layer_signals["ui"] and layer_signals["data"] and not layer_signals["domain"]:
        project_structure.append(
            "UI and data layers are visible but a domain/use-case layer is not. Add `domain/` use-case boundaries so UI does not orchestrate business logic directly."
        )
    if len(utils_files) >= 4:
        project_structure.append(
            f"`utils` appears overloaded ({len(utils_files)} files). Split by concern (e.g., `time/`, `validation/`, `platform/`) to avoid a dumping ground."
        )
    if modules and len(modules) <= 1 and has_android:
        project_structure.append(
            "Single-module Android setup detected. Introduce feature or core modules to enforce one-direction dependency flow."
        )
    if large_code_files:
        biggest = large_code_files[0]
        project_structure.append(
            f"Large file detected ({biggest[1]} lines): `{biggest[0]}`. Break it into smaller units before adding new features."
        )
    if not project_structure:
        project_structure.append(
            "Current package split looks coherent; preserve one-direction dependency flow with module boundaries or dependency checks."
        )

    build_run = []
    if not has_run:
        build_run.append(
            "README run/setup instructions are missing or weak. Add a 2-minute quickstart with exact commands and required tool versions."
        )
    if has_gradle_files and not has_gradlew:
        build_run.append(
            "Gradle wrapper is missing. Commit `gradlew` + wrapper files so contributors can run builds with a pinned Gradle version."
        )
    if not has_env_example and has_runtime_config and has_network_markers:
        build_run.append(
            "Project appears to use network/API flows but no `.env.example` detected. Add sample config keys and required defaults."
        )
    if not has_ci:
        build_run.append(
            "No CI workflow detected. Add a pipeline that runs build, unit tests, lint, and basic static checks on pull requests."
        )
    if not build_run:
        build_run.append(
            "Build/run onboarding looks healthy; keep setup docs and CI checks aligned with each release."
        )

    code_quality = []
    if not has_tests:
        code_quality.append(
            "No test footprint detected. Add smoke tests first, then cover auth/cart/checkout flows with focused regression tests."
        )
    if has_tests and not has_ui_tests and has_android:
        code_quality.append(
            "Unit tests may exist but UI/instrumentation coverage is not visible. Add Android UI tests for navigation and critical user journeys."
        )
    if not has_lint:
        if has_android:
            code_quality.append(
                "No lint/format config detected. Add detekt + ktlint (or equivalent) and enforce in CI."
            )
        else:
            code_quality.append(
                "No lint/format config detected. Add language-appropriate linting/formatting rules and enforce in CI."
            )
    if re.search(r"\bprint\(|\bprintln\(|system\.out\.print", context_lower):
        code_quality.append(
            "Raw print statements detected in scanned context. Replace with structured logging and standardized tags."
        )
    if not code_quality:
        code_quality.append(
            "Code quality signals are acceptable; next step is to tighten regression coverage around high-change files."
        )

    security_reliability = []
    if secret_hits:
        security_reliability.append(
            "Potential hardcoded secret pattern found in scanned context. Move secrets to secure config and rotate exposed values."
        )
    if permissions:
        risky_permissions = [
            p
            for p in permissions
            if any(
                risky in p
                for risky in (
                    "READ_SMS",
                    "RECEIVE_SMS",
                    "READ_CONTACTS",
                    "ACCESS_FINE_LOCATION",
                    "READ_PHONE_STATE",
                    "WRITE_EXTERNAL_STORAGE",
                )
            )
        ]
        if risky_permissions:
            security_reliability.append(
                "Sensitive Android permissions detected ("
                + ", ".join(sorted(set(risky_permissions)))
                + "). Re-validate least-privilege and user-facing justification."
            )
    if has_android and not has_ui_tests and (
        any("auth" in path for path in index_lower) or any("checkout" in path for path in index_lower)
    ):
        security_reliability.append(
            "Auth/checkout surfaces exist without visible UI regression gates. Add reliability tests before shipping high-risk flows."
        )
    if not security_reliability:
        security_reliability.append(
            "No high-severity security or reliability smell detected from current scan; continue with secret scanning and permission review in CI."
        )

    performance_scalability = []
    if has_android and not has_benchmark:
        performance_scalability.append(
            "No benchmark module detected. Add macrobenchmark tests for startup and critical screen transitions."
        )
    if has_android and not has_baseline_profile:
        performance_scalability.append(
            "No baseline profile setup found. Add baseline profiles to improve startup and scrolling performance on real devices."
        )
    if has_network_markers and not has_cache_markers:
        performance_scalability.append(
            "Network-heavy signals detected without obvious caching markers. Add cache strategy (TTL/invalidation) for high-traffic reads."
        )
    if has_db_markers and "index" not in context_lower:
        performance_scalability.append(
            "Database usage is visible but index strategy is not. Review hot queries and add index checks to avoid hidden latency spikes."
        )
    if not performance_scalability:
        performance_scalability.append(
            "Performance baseline looks reasonable from scanned context; keep profiling and benchmark budgets in the CI loop."
        )

    product_thinking = []
    if not has_target_user:
        product_thinking.append(
            "README does not clearly state target users. Add a 1-2 sentence audience statement near the top."
        )
    if not has_differentiator:
        product_thinking.append(
            "Differentiator is not explicit. Add a short 'Why this over alternatives' section with concrete tradeoffs."
        )
    if not has_roadmap:
        product_thinking.append(
            "No roadmap/future features section found. Add near-term milestones and validation criteria."
        )
    if not has_demo:
        product_thinking.append(
            "Product credibility is limited without demo artifacts. Add screenshots/video for core flows and keep them versioned."
        )
    if not product_thinking:
        product_thinking.append(
            "Product framing is already clear; next leverage point is adding measurable success metrics per roadmap item."
        )

    sections = [
        FounderAuditSection("First impression (10 seconds)", first_impression),
        FounderAuditSection("Project structure (architecture)", project_structure),
        FounderAuditSection("Build & run experience", build_run),
        FounderAuditSection("Code quality signals", code_quality),
        FounderAuditSection("Security & reliability", security_reliability),
        FounderAuditSection("Performance & scalability", performance_scalability),
        FounderAuditSection("Product-level thinking (founder mode)", product_thinking),
    ]

    return FounderAuditReport(
        sections=sections,
        analyzed_files=len(repo_index),
        index_truncated=truncated,
    )


def build_scoring_config(custom: dict | None = None) -> dict:
    config = {
        "action_base_scores": dict(DEFAULT_ACTION_SCORES),
        "signal_patterns": dict(DEFAULT_SIGNAL_PATTERNS),
        "signal_boosts": {
            signal: dict(boosts) for signal, boosts in DEFAULT_SIGNAL_BOOSTS.items()
        },
        "signal_thresholds": dict(DEFAULT_SIGNAL_THRESHOLDS),
    }
    if not custom:
        return config

    raw_actions = custom.get("action_base_scores", {})
    if isinstance(raw_actions, dict):
        for action, score in raw_actions.items():
            try:
                config["action_base_scores"][str(action)] = float(score)
            except (TypeError, ValueError):
                continue

    raw_patterns = custom.get("signal_patterns", {})
    if isinstance(raw_patterns, dict):
        for signal, pattern in raw_patterns.items():
            if isinstance(pattern, str):
                config["signal_patterns"][str(signal)] = pattern

    raw_boosts = custom.get("signal_boosts", {})
    if isinstance(raw_boosts, dict):
        for signal, boost_map in raw_boosts.items():
            if not isinstance(boost_map, dict):
                continue
            signal_key = str(signal)
            current = dict(config["signal_boosts"].get(signal_key, {}))
            for action, delta in boost_map.items():
                try:
                    current[str(action)] = float(delta)
                except (TypeError, ValueError):
                    continue
            config["signal_boosts"][signal_key] = current

    raw_thresholds = custom.get("signal_thresholds", {})
    if isinstance(raw_thresholds, dict):
        for signal, threshold in raw_thresholds.items():
            try:
                config["signal_thresholds"][str(signal)] = max(float(threshold), 0.01)
            except (TypeError, ValueError):
                continue

    return config


def load_scoring_config_file(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"file not found: {path}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"read failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("top-level json value must be an object")
    return build_scoring_config(payload)


def build_prediction_payload(
    report: PredictionReport,
    include_scan: bool,
    scan: dict,
    founder_audit: FounderAuditReport | None = None,
) -> dict:
    payload = {
        "likely_next_actions": [item.__dict__ for item in report.likely_next_actions],
        "current_intent_inference": report.current_intent_inference,
        "decision_predictions": report.decision_predictions,
        "future_improvements": [item.__dict__ for item in report.future_improvements],
        "style_deviations_detected": report.style_deviations_detected,
        "reasoning_trace": report.reasoning_trace,
        "alternative_paths": [item.__dict__ for item in report.alternative_paths],
    }
    if founder_audit is not None:
        payload["founder_audit"] = founder_audit.to_dict()
    if include_scan:
        payload["scan"] = scan
    return payload


def build_snapshot_document(
    prediction_payload: dict,
    context: str,
    snapshot_tag: str = "",
    weights_file: str = "",
    show_scan: bool = False,
) -> dict:
    context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "generated_at_utc": (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        ),
        "tag": snapshot_tag.strip(),
        "context_sha256": context_hash,
        "context_length_chars": len(context),
        "run_config": {
            "weights_file": weights_file.strip(),
            "show_scan": bool(show_scan),
        },
        "prediction": prediction_payload,
    }


def write_snapshot_file(path: str, snapshot: dict) -> str:
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to write snapshot: {exc}") from exc
    return str(file_path)


def resolve_project_path(project_dir: str, raw_path: str) -> str:
    base = Path(project_dir).expanduser().resolve()
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


def _coerce_threshold(raw_value: object, fallback: float) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return fallback
    return max(value, 0.01)


def _context_source_type_for_path(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    ext = Path(normalized).suffix

    if (
        "/test/" in normalized
        or "/tests/" in normalized
        or name.endswith("_test.py")
        or name.endswith("test.kt")
        or name.endswith("test.java")
    ):
        return "test"
    if ext in LOG_EXTENSIONS:
        return "log"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in DOC_EXTENSIONS:
        return "doc"
    if ext in SOURCE_CODE_EXTENSIONS:
        return "source"
    if ext in CONFIG_EXTENSIONS:
        return "config"
    return "other_file"


def _parse_context_sections(context: str) -> List[ContextSection]:
    if not context.strip():
        return []

    file_marker_pattern = re.compile(r"^\[file:(.+)\]\s*$")
    git_marker_pattern = re.compile(r"^\[git-log:last-\d+-commits\]\s*$", flags=re.IGNORECASE)
    sections: List[ContextSection] = []
    buffer: List[str] = []
    current_source_type = "inline"
    current_source_ref = "inline"
    current_weight = DEFAULT_CONTEXT_SOURCE_WEIGHTS["inline"]

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if not text:
            return
        sections.append(
            ContextSection(
                source_type=current_source_type,
                source_ref=current_source_ref,
                weight=current_weight,
                text=text,
            )
        )

    for raw_line in context.splitlines():
        file_match = file_marker_pattern.match(raw_line)
        if file_match:
            flush()
            buffer = []
            path = file_match.group(1).strip()
            source_type = _context_source_type_for_path(path)
            current_source_type = source_type
            current_source_ref = path
            current_weight = DEFAULT_CONTEXT_SOURCE_WEIGHTS[source_type]
            continue

        if git_marker_pattern.match(raw_line):
            flush()
            buffer = []
            current_source_type = "git-log"
            current_source_ref = "git-log"
            current_weight = DEFAULT_CONTEXT_SOURCE_WEIGHTS["git-log"]
            continue

        buffer.append(raw_line)

    flush()
    return sections


def _weighted_pattern_score(
    pattern: str,
    sections: List[ContextSection],
    allowed_source_types: set[str] | None = None,
) -> float:
    if not pattern:
        return 0.0
    score = 0.0
    for section in sections:
        if allowed_source_types is not None and section.source_type not in allowed_source_types:
            continue
        if re.search(pattern, section.text.lower()):
            score += section.weight
    return score


def _android_path_score(sections: List[ContextSection]) -> float:
    score = 0.0
    for section in sections:
        if not section.source_ref or section.source_ref in {"inline", "git-log"}:
            continue
        path = section.source_ref.replace("\\", "/").lower()
        if re.search(r"androidmanifest\.xml|build\.gradle(\.kts)?|settings\.gradle(\.kts)?", path):
            score += 0.50
            continue
        if "/androidtest/" in path:
            score += 0.45
            continue
        if path.endswith(".kt") or path.endswith(".java"):
            score += 0.35
            continue
        if "android" in path:
            score += 0.25
    return min(score, 1.40)


class DigitalTwinPredictor:
    """Deterministic, evidence-based predictor for technical work patterns."""

    def __init__(self, baseline: dict | None = None, scoring_config: dict | None = None):
        self.baseline = baseline or BASELINE_PATTERNS
        self.scoring_config = build_scoring_config(scoring_config)

    def predict(self, context: str) -> PredictionReport:
        ctx = context.strip()
        sections = _parse_context_sections(ctx)

        action_scores = dict(self.scoring_config["action_base_scores"])

        reasoning = []
        deviations = []
        decisions = []

        # Signal extraction from context text with source-aware confidence weighting.
        signal_thresholds = self.scoring_config.get("signal_thresholds", {})
        signal_hits = {}
        for signal, pattern in self.scoring_config["signal_patterns"].items():
            if not pattern:
                signal_hits[signal] = False
                continue
            threshold = _coerce_threshold(
                signal_thresholds.get(
                    signal,
                    DEFAULT_SIGNAL_THRESHOLDS.get(signal, 0.85),
                ),
                DEFAULT_SIGNAL_THRESHOLDS.get(signal, 0.85),
            )
            score = _weighted_pattern_score(pattern, sections)
            signal_hits[signal] = score >= threshold

        for signal, hit in signal_hits.items():
            if not hit:
                continue
            for action, delta in self.scoring_config["signal_boosts"].get(signal, {}).items():
                action_scores[action] = action_scores.get(action, 0.0) + float(delta)

        has_benchmark = signal_hits.get("benchmark", False)
        has_bug = signal_hits.get("bug", False)
        has_arch = signal_hits.get("arch", False)
        has_blocked = signal_hits.get("blocked", False)
        has_framework = signal_hits.get("framework", False)
        has_long_horizon = signal_hits.get("long_horizon", False)
        has_logs = signal_hits.get("logs", False)
        has_db = signal_hits.get("db", False)
        high_signal_sources = {"inline", "git-log", "log", "text"}
        narrative_sources = high_signal_sources | {"doc", "config", "other_file"}
        has_git_history = any(section.source_type == "git-log" for section in sections)
        has_recent_delivery_markers = (
            _weighted_pattern_score(
                EXTRA_SIGNAL_PATTERNS["delivery_markers"],
                sections,
                allowed_source_types=high_signal_sources,
            )
            >= _coerce_threshold(
                signal_thresholds.get(
                    "delivery_markers",
                    DEFAULT_EXTRA_SIGNAL_THRESHOLDS["delivery_markers"],
                ),
                DEFAULT_EXTRA_SIGNAL_THRESHOLDS["delivery_markers"],
            )
        )
        has_android = (
            _weighted_pattern_score(
                EXTRA_SIGNAL_PATTERNS["android"],
                sections,
                allowed_source_types=narrative_sources,
            )
            + _android_path_score(sections)
            >= _coerce_threshold(
                signal_thresholds.get("android", DEFAULT_EXTRA_SIGNAL_THRESHOLDS["android"]),
                DEFAULT_EXTRA_SIGNAL_THRESHOLDS["android"],
            )
        )
        has_ui_flows = (
            _weighted_pattern_score(
                EXTRA_SIGNAL_PATTERNS["ui_flows"],
                sections,
                allowed_source_types=narrative_sources,
            )
            >= _coerce_threshold(
                signal_thresholds.get("ui_flows", DEFAULT_EXTRA_SIGNAL_THRESHOLDS["ui_flows"]),
                DEFAULT_EXTRA_SIGNAL_THRESHOLDS["ui_flows"],
            )
        )
        has_framework_overreach = (
            _weighted_pattern_score(
                EXTRA_SIGNAL_PATTERNS["framework_overreach"],
                sections,
                allowed_source_types=narrative_sources,
            )
            >= _coerce_threshold(
                signal_thresholds.get(
                    "framework_overreach",
                    DEFAULT_EXTRA_SIGNAL_THRESHOLDS["framework_overreach"],
                ),
                DEFAULT_EXTRA_SIGNAL_THRESHOLDS["framework_overreach"],
            )
        )

        if has_benchmark:
            decisions.append("Prefer measuring before rewriting; profile first, then optimize the top hotspot only.")
            reasoning.append("Performance terms in context indicate optimization-first sequencing.")

        if has_bug:
            decisions.append("Constrain blast radius with minimal, test-backed fixes before broader refactors.")
            reasoning.append("Failure signals imply immediate containment and reproducibility work.")

        if has_arch:
            decisions.append("Keep architecture artifacts current to preserve long-term correctness under iteration.")
            reasoning.append("Architecture references align with design-first behavior.")

        if has_blocked:
            decisions.append("Pivot to an unblocked subsystem while capturing exact blocker state for fast return.")
            reasoning.append("Blocker language maps to strategic task switching rather than idle context switching.")

        if has_framework:
            if has_android and not has_framework_overreach:
                reasoning.append(
                    "Framework terms appeared in Android context without over-engineering markers; treated as neutral."
                )
            else:
                deviations.append(
                    "Context includes framework-heavy language; baseline style usually favors lighter-weight, purpose-fit abstractions."
                )
                reasoning.append("Potential mismatch detected: framework reliance vs minimal-framework preference.")

        if has_long_horizon:
            decisions.append("Bias toward modular boundaries and explicit interfaces for long-horizon maintainability.")
            reasoning.append("Long-horizon signals increase emphasis on architecture and interface stability.")

        if has_logs:
            decisions.append("Use low-level traces/log correlation before changing core logic to avoid blind edits.")
            reasoning.append("Trace/log evidence suggests root-cause-first debugging path.")

        if has_db:
            decisions.append("Evaluate query/index tradeoffs before scaling app-layer complexity.")
            reasoning.append("Storage/query terms imply system-level bottleneck analysis.")

        if has_android:
            reasoning.append("Android project signals detected (Kotlin/Gradle/Compose/ViewModel patterns).")
            for action, base in ANDROID_ACTION_PRIORS.items():
                action_scores[action] = action_scores.get(action, 0.0) + base
            if has_ui_flows:
                action_scores[
                    "Add instrumentation/UI tests for critical user journeys (auth, cart, checkout)"
                ] += 0.24
            if has_benchmark:
                action_scores[
                    "Profile startup and scroll performance on representative Android devices"
                ] += 0.20
            if has_bug:
                action_scores[
                    "Harden ViewModel state and side-effect handling for Compose screens"
                ] += 0.16
            decisions.append("Treat startup, navigation, and state-management regressions as first-class mobile quality risks.")
            if has_ui_flows:
                decisions.append("Protect cart/checkout/auth flows with UI tests before broad feature expansion.")

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
        future_improvements = self._future_improvements(
            has_benchmark=has_benchmark,
            has_bug=has_bug,
            has_arch=has_arch,
            has_blocked=has_blocked,
            has_db=has_db,
            has_long_horizon=has_long_horizon,
            has_git_history=has_git_history,
            has_recent_delivery_markers=has_recent_delivery_markers,
            has_android=has_android,
            has_ui_flows=has_ui_flows,
        )

        return PredictionReport(
            likely_next_actions=ranked,
            current_intent_inference=intent,
            decision_predictions=decisions,
            future_improvements=future_improvements,
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
        if has_benchmark:
            return (
                "High probability objective: isolate the dominant performance bottleneck and improve throughput with measured, low-risk changes."
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

    @staticmethod
    def _future_improvements(
        has_benchmark: bool,
        has_bug: bool,
        has_arch: bool,
        has_blocked: bool,
        has_db: bool,
        has_long_horizon: bool,
        has_git_history: bool,
        has_recent_delivery_markers: bool,
        has_android: bool,
        has_ui_flows: bool,
    ) -> List[RankedItem]:
        improvements = [
            RankedItem(
                title="Expand regression and benchmark coverage for recently touched hotspots",
                probability=round(
                    min(
                        0.95,
                        0.52 + (0.20 if has_bug else 0.0) + (0.16 if has_benchmark else 0.0),
                    ),
                    2,
                ),
                rationale="Recent reliability/performance signals indicate the highest ROI comes from repeatable validation around changed paths.",
            ),
            RankedItem(
                title="Introduce CI guardrails (tests, lint, and performance budgets) for change gating",
                probability=round(
                    min(
                        0.95,
                        0.50 + (0.14 if (has_bug or has_benchmark) else 0.0),
                    ),
                    2,
                ),
                rationale="Automated gates reduce regression risk and preserve behavior as iteration speed increases.",
            ),
            RankedItem(
                title="Create or update ADRs and module boundaries before the next major refactor",
                probability=round(
                    min(
                        0.95,
                        0.46 + (0.18 if has_arch else 0.0) + (0.12 if has_long_horizon else 0.0),
                    ),
                    2,
                ),
                rationale="Architecture discipline compounds over long-running projects and lowers integration risk.",
            ),
            RankedItem(
                title="Automate changelog/release-note generation from commit history",
                probability=round(
                    min(
                        0.95,
                        0.38
                        + (0.16 if has_git_history else 0.0)
                        + (0.10 if has_recent_delivery_markers else 0.0),
                    ),
                    2,
                ),
                rationale="If delivery history is active, automated summarization improves traceability and planning continuity.",
            ),
            RankedItem(
                title="Harden storage observability with query plan checks and index review cadence",
                probability=round(
                    min(0.95, 0.40 + (0.24 if has_db else 0.0)),
                    2,
                ),
                rationale="Database-linked work benefits from explicit telemetry and index hygiene to prevent hidden bottlenecks.",
            ),
            RankedItem(
                title="Maintain a blocker-handling queue for parallelizable tasks",
                probability=round(
                    min(0.95, 0.36 + (0.24 if has_blocked else 0.0)),
                    2,
                ),
                rationale="When blockers recur, a prepared fallback queue protects delivery velocity without context loss.",
            ),
            RankedItem(
                title="Derive a technical-debt roadmap from repeated fix/refactor patterns",
                probability=round(
                    min(
                        0.95,
                        0.42 + (0.16 if has_git_history else 0.0) + (0.08 if has_bug else 0.0),
                    ),
                    2,
                ),
                rationale="Historical change patterns expose structural debt clusters that are best addressed intentionally.",
            ),
        ]

        if has_android:
            improvements.extend(
                [
                    RankedItem(
                        title="Add macrobenchmark + baseline profile coverage for startup and critical rendering paths",
                        probability=round(
                            min(0.95, 0.56 + (0.16 if has_benchmark else 0.0)),
                            2,
                        ),
                        rationale="Android UX quality improves significantly when startup and jank are tracked with device-level benchmarks.",
                    ),
                    RankedItem(
                        title="Strengthen Compose UI testing for navigation and purchase funnels",
                        probability=round(
                            min(0.95, 0.54 + (0.18 if has_ui_flows else 0.0)),
                            2,
                        ),
                        rationale="Instrumented UI tests catch regressions in high-value flows before release.",
                    ),
                    RankedItem(
                        title="Standardize ViewModel state contracts and one-off event handling",
                        probability=round(
                            min(0.95, 0.50 + (0.12 if has_bug else 0.0)),
                            2,
                        ),
                        rationale="Consistent UI-state modeling reduces flaky behavior and makes debugging faster in Compose apps.",
                    ),
                    RankedItem(
                        title="Add release-track crash and ANR monitoring gates",
                        probability=round(
                            min(0.95, 0.48 + (0.14 if has_recent_delivery_markers else 0.0)),
                            2,
                        ),
                        rationale="Mobile delivery cadence benefits from release health gates that block risky rollouts early.",
                    ),
                ]
            )

        return sorted(improvements, key=lambda x: x.probability, reverse=True)[:5]


def _skip(path: str, reason: str) -> dict:
    severity = SKIP_REASON_SEVERITY.get(reason, "low")
    hint = SKIP_REASON_HINTS.get(reason, "Review scan filters and file accessibility.")
    return {
        "path": path,
        "reason": reason,
        "severity": severity,
        "severity_score": SEVERITY_RANK[severity],
        "hint": hint,
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


def summarize_remediation_hints(skipped_files: List[dict]) -> List[dict]:
    grouped = {}
    for item in skipped_files:
        reason = item.get("reason", "unknown")
        if reason in grouped:
            grouped[reason]["count"] += 1
            continue
        severity = item.get("severity", SKIP_REASON_SEVERITY.get(reason, "low"))
        grouped[reason] = {
            "reason": reason,
            "severity": severity,
            "hint": item.get("hint", SKIP_REASON_HINTS.get(reason, "Review scan filters and file accessibility.")),
            "count": 1,
        }
    return sorted(
        grouped.values(),
        key=lambda item: (
            -SEVERITY_RANK.get(item["severity"], 1),
            -item["count"],
            item["reason"],
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


def gather_git_context(max_commits: int = 8, repo_dir: str | None = None) -> str:
    cmd = ["git"]
    if repo_dir:
        cmd.extend(["-C", repo_dir])
    cmd.extend(
        [
            "log",
            f"--max-count={max_commits}",
            "--pretty=format:%h %ad %s",
            "--date=short",
        ]
    )
    try:
        result = subprocess.run(
            cmd,
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


def gather_git_changed_files(max_commits: int = 20, repo_dir: str | None = None) -> List[str]:
    cmd = ["git"]
    if repo_dir:
        cmd.extend(["-C", repo_dir])
    cmd.extend(
        [
            "log",
            f"--max-count={max_commits}",
            "--name-only",
            "--pretty=format:",
        ]
    )
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []

    if result.returncode != 0:
        return []

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return []

    base = Path(repo_dir).resolve() if repo_dir else None
    files = []
    seen = set()
    for raw in lines:
        path = Path(raw)
        if base and not path.is_absolute():
            path = base / path
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        files.append(key)
    return files


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
    git_file_focus_limit: int = 30,
    git_repo_dir: str | None = None,
) -> tuple[str, dict]:
    parts = []
    scan = {
        "included_files": [],
        "skipped_files": [],
        "skip_summary": {"by_reason": {}, "by_severity": {"high": 0, "medium": 0, "low": 0}},
        "remediation_hints": [],
        "git_focus_files": 0,
        "from_git": {"requested": from_git, "included": False, "reason": ""},
    }

    clean_inline = inline_context.strip()
    if clean_inline:
        parts.append(clean_inline)

    candidate_files = []
    dedupe_seen = set()
    if from_git and git_file_focus_limit > 0:
        scope_dir = Path(from_dir).resolve() if from_dir else None
        repo_base = Path(git_repo_dir).resolve() if git_repo_dir else None
        ignore_patterns = dir_ignore_patterns or []
        ignore_dirs_set = set(dir_ignore_dirs or [])
        active_patterns = dir_patterns or []
        focused = gather_git_changed_files(
            max_commits=max(git_commits, 1),
            repo_dir=git_repo_dir,
        )[: max(git_file_focus_limit, 1)]
        accepted_focus = 0
        for raw in focused:
            path = Path(raw)
            key = str(path)
            if not path.exists() or not path.is_file():
                scan["skipped_files"].append(_skip(key, "missing_or_not_file"))
                continue

            rel_for_filters = path.as_posix()
            rel_parts = path.parts
            if repo_base:
                try:
                    rel_obj = path.relative_to(repo_base)
                    rel_for_filters = rel_obj.as_posix()
                    rel_parts = rel_obj.parts
                except ValueError:
                    pass

            if scope_dir:
                try:
                    path.relative_to(scope_dir)
                except ValueError:
                    scan["skipped_files"].append(_skip(key, "outside_from_dir"))
                    continue

            if any(part in ignore_dirs_set for part in rel_parts):
                scan["skipped_files"].append(_skip(key, "ignored_dir"))
                continue

            if ignore_patterns and (
                any(fnmatch.fnmatch(path.name, p) for p in ignore_patterns)
                or any(fnmatch.fnmatch(rel_for_filters, p) for p in ignore_patterns)
            ):
                scan["skipped_files"].append(_skip(key, "ignored_pattern"))
                continue

            if active_patterns and not (
                any(fnmatch.fnmatch(path.name, p) for p in active_patterns)
                or any(fnmatch.fnmatch(rel_for_filters, p) for p in active_patterns)
            ):
                scan["skipped_files"].append(_skip(key, "outside_dir_patterns"))
                continue

            try:
                stat = path.stat()
            except OSError:
                scan["skipped_files"].append(_skip(key, "stat_error"))
                continue
            if stat.st_size > max(dir_max_bytes, 1):
                scan["skipped_files"].append(_skip(key, "over_max_bytes"))
                continue
            if key in dedupe_seen:
                continue
            dedupe_seen.add(key)
            candidate_files.append(key)
            accepted_focus += 1
        scan["git_focus_files"] = accepted_focus

    if from_files:
        for raw in from_files:
            key = str(Path(raw))
            if key in dedupe_seen:
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
        git_ctx = gather_git_context(max_commits=git_commits, repo_dir=git_repo_dir)
        if git_ctx:
            scan["from_git"]["included"] = True
            parts.append(git_ctx)
        else:
            scan["from_git"]["reason"] = "git_unavailable_or_no_history"

    scan["skipped_files"] = sort_skips_by_severity(scan["skipped_files"])
    scan["skip_summary"] = summarize_skips(scan["skipped_files"])
    scan["remediation_hints"] = summarize_remediation_hints(scan["skipped_files"])

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
    git_file_focus_limit: int = 30,
    git_repo_dir: str | None = None,
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
        git_file_focus_limit=git_file_focus_limit,
        git_repo_dir=git_repo_dir,
    )
    return context


def format_scan_metadata(scan: dict) -> str:
    lines = []
    skip_summary = scan.get("skip_summary", summarize_skips(scan["skipped_files"]))
    skipped_files = sort_skips_by_severity(scan["skipped_files"])
    remediation_hints = scan.get("remediation_hints", summarize_remediation_hints(skipped_files))

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
    if scan.get("git_focus_files", 0):
        lines.append(f"5. Git-focused files injected: {scan['git_focus_files']}")

    if skip_summary["by_reason"]:
        lines.append("\nTop Skip Reasons:")
        top_reasons = sorted(
            skip_summary["by_reason"].items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        for reason, count in top_reasons:
            sev = SKIP_REASON_SEVERITY.get(reason, "low")
            lines.append(f"- {reason}: {count} ({sev})")

    if remediation_hints:
        lines.append("\nRecommended Remediations:")
        for item in remediation_hints[:5]:
            lines.append(
                f"- {item['reason']} ({item['severity']}, count={item['count']}): {item['hint']}"
            )

    if scan["included_files"]:
        lines.append("\nIncluded File Paths:")
        for path in scan["included_files"]:
            lines.append(f"- {path}")

    if skipped_files:
        lines.append("\nSkipped File Paths:")
        for item in skipped_files:
            lines.append(
                f"- {item['path']} [{item['reason']}] severity={item['severity']} hint={item['hint']}"
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
        "--project-dir",
        type=str,
        default=".",
        help="Target project directory used for relative paths and --from-git.",
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
        "--git-file-focus-limit",
        type=int,
        default=30,
        help="When using --from-git, prioritize this many recently changed files into context.",
    )
    parser.add_argument(
        "--show-scan",
        action="store_true",
        help="Show metadata for ingested and skipped files.",
    )
    parser.add_argument(
        "--founder-audit",
        action="store_true",
        help="Add a 7-part project improvement audit (first impression, architecture, quality, security, performance, product).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print only the founder audit section (skip prediction markdown output).",
    )
    parser.add_argument(
        "--audit-max-files",
        type=int,
        default=5000,
        help="Max files indexed for founder audit heuristics.",
    )
    parser.add_argument(
        "--weights-file",
        type=str,
        default="",
        help="JSON scoring config overrides (action scores, signal patterns, boosts).",
    )
    parser.add_argument(
        "--snapshot-out",
        type=str,
        default="",
        help="Write a JSON snapshot artifact (prediction + metadata) to this path.",
    )
    parser.add_argument(
        "--snapshot-tag",
        type=str,
        default="",
        help="Optional label included in snapshot metadata.",
    )

    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists() or not project_dir.is_dir():
        raise SystemExit(f"Invalid --project-dir: {project_dir} is not a directory")

    resolved_from_files = [
        resolve_project_path(str(project_dir), raw) for raw in args.from_files
    ]
    resolved_from_dir = (
        resolve_project_path(str(project_dir), args.from_dir.strip())
        if args.from_dir.strip()
        else None
    )

    context, scan = resolve_context_with_metadata(
        inline_context=args.context,
        from_files=resolved_from_files,
        from_dir=resolved_from_dir,
        dir_patterns=args.dir_patterns,
        dir_max_files=max(args.dir_max_files, 1),
        dir_recursive=not args.non_recursive,
        dir_ignore_patterns=args.ignore_patterns,
        dir_max_bytes=max(args.dir_max_bytes, 1),
        dir_ignore_dirs=args.ignore_dirs,
        from_git=args.from_git,
        git_commits=max(args.git_commits, 1),
        git_file_focus_limit=max(args.git_file_focus_limit, 0),
        git_repo_dir=str(project_dir),
    )
    if not context:
        context = input("Paste context: ").strip()

    scoring_config = None
    if args.weights_file.strip():
        try:
            scoring_config = load_scoring_config_file(
                resolve_project_path(str(project_dir), args.weights_file.strip())
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid --weights-file: {exc}") from exc

    founder_audit_report = None
    if args.founder_audit or args.audit_only:
        founder_audit_report = build_founder_audit_report(
            project_dir=str(project_dir),
            context=context,
            scan=scan,
            ignore_dirs=args.ignore_dirs,
            max_files=max(args.audit_max_files, 1),
        )

    predictor = DigitalTwinPredictor(scoring_config=scoring_config)
    report = predictor.predict(context)
    payload = build_prediction_payload(
        report=report,
        include_scan=args.show_scan,
        scan=scan,
        founder_audit=founder_audit_report,
    )

    if args.json:
        print(
            json.dumps(
                payload,
                indent=2,
            )
        )
    else:
        if not args.audit_only:
            print(report.to_markdown())
        if founder_audit_report is not None:
            if not args.audit_only:
                print()
            print(founder_audit_report.to_markdown())
        if args.show_scan:
            print()
            print(format_scan_metadata(scan))

    if args.snapshot_out.strip():
        snapshot = build_snapshot_document(
            prediction_payload=payload,
            context=context,
            snapshot_tag=args.snapshot_tag,
            weights_file=args.weights_file,
            show_scan=args.show_scan,
        )
        try:
            output_path = write_snapshot_file(
                resolve_project_path(str(project_dir), args.snapshot_out.strip()),
                snapshot,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid --snapshot-out: {exc}") from exc
        print(f"snapshot written: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
