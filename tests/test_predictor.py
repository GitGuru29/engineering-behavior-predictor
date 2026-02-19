import unittest
import tempfile
import json
import hashlib
from pathlib import Path
import os
import textwrap

from src.predictor import (
    build_founder_audit_report,
    build_prediction_payload,
    build_snapshot_document,
    DigitalTwinPredictor,
    discover_context_files,
    discover_context_files_with_metadata,
    format_scan_metadata,
    gather_git_changed_files,
    gather_git_context,
    load_scoring_config_file,
    resolve_project_path,
    resolve_context,
    resolve_context_with_metadata,
    write_snapshot_file,
)


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.p = DigitalTwinPredictor()

    def test_bug_and_perf_intent(self):
        context = "Regression after cache rewrite. Latency increased and error traces show timeout failures."
        report = self.p.predict(context)
        self.assertIn("stabilize correctness", report.current_intent_inference.lower())
        self.assertEqual(len(report.likely_next_actions), 6)
        self.assertGreaterEqual(len(report.future_improvements), 3)

    def test_future_improvements_include_release_automation_when_git_context_present(self):
        context = "[git-log:last-8-commits]\nabc123 2026-02-10 feat: add parser\nbcd234 2026-02-11 fix: db timeout"
        report = self.p.predict(context)
        titles = [item.title.lower() for item in report.future_improvements]
        self.assertTrue(any("changelog/release-note generation" in t for t in titles))

    def test_android_context_produces_android_specific_improvements(self):
        context = """
        AndroidManifest.xml
        app/build.gradle.kts
        MainActivity.kt
        ViewModel
        Compose screen navigation checkout cart auth
        """
        report = self.p.predict(context)
        titles = [item.title.lower() for item in report.future_improvements]
        self.assertTrue(any("compose ui testing" in t for t in titles))
        self.assertTrue(any("baseline profile" in t for t in titles))

    def test_framework_deviation_detection(self):
        context = "We should add another framework and boilerplate-heavy stack."
        report = self.p.predict(context)
        self.assertTrue(report.style_deviations_detected)

    def test_library_word_alone_not_flagged_as_framework_deviation(self):
        context = "Using Kotlin stdlib and a shared utility library."
        report = self.p.predict(context)
        self.assertFalse(report.style_deviations_detected)

    def test_android_framework_terms_without_overengineering_are_neutral(self):
        context = "Android framework components in MainActivity.kt and Compose navigation."
        report = self.p.predict(context)
        self.assertFalse(report.style_deviations_detected)

    def test_architecture_signal(self):
        context = "Need an ADR and architecture update for long-term roadmap."
        report = self.p.predict(context)
        top_titles = [x.title.lower() for x in report.likely_next_actions[:2]]
        self.assertTrue(any("architecture/design note" in t for t in top_titles))

    def test_source_and_docs_noise_do_not_trigger_unrelated_signals(self):
        context = textwrap.dedent(
            """
            [file:/repo/src/predictor.py]
            DEFAULT_SIGNAL_PATTERNS = {"framework": r"\\bframework\\b|boilerplate", "bug": "bug|error"}

            [file:/repo/tests/test_predictor.py]
            def test_android_context():
                context = "AndroidManifest.xml Compose checkout cart auth"

            [file:/repo/README.md]
            architecture roadmap framework blockers and dependency notes

            [file:/repo/runtime.log]
            latency spike observed on p95 endpoint
            """
        )
        report = self.p.predict(context)
        self.assertFalse(report.style_deviations_detected)
        self.assertIn("performance bottleneck", report.current_intent_inference.lower())
        self.assertFalse(any("blast radius" in d.lower() for d in report.decision_predictions))
        self.assertFalse(any("mobile quality risks" in d.lower() for d in report.decision_predictions))

    def test_android_detection_from_file_path_markers(self):
        context = textwrap.dedent(
            """
            [file:/repo/app/src/main/AndroidManifest.xml]
            <manifest package="com.example.app"></manifest>

            [file:/repo/app/src/main/java/com/example/MainActivity.kt]
            class MainActivity
            """
        )
        report = self.p.predict(context)
        self.assertTrue(any("mobile quality risks" in d.lower() for d in report.decision_predictions))
        titles = [item.title.lower() for item in report.future_improvements]
        self.assertTrue(any("baseline profile" in title for title in titles))

    def test_custom_signal_thresholds_can_reduce_noise(self):
        predictor = DigitalTwinPredictor(scoring_config={"signal_thresholds": {"bug": 2.0}})
        report = predictor.predict("regression and bug in checkout flow")
        self.assertFalse(any("blast radius" in d.lower() for d in report.decision_predictions))

    def test_resolve_context_from_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = Path(tmp) / "notes.txt"
            f2 = Path(tmp) / "logs.txt"
            f1.write_text("Architecture ADR pending", encoding="utf-8")
            f2.write_text("Latency regression in service", encoding="utf-8")

            merged = resolve_context(
                inline_context="",
                from_files=[str(f1), str(f2)],
                from_git=False,
            )

            self.assertIn("[file:", merged)
            self.assertIn("Architecture ADR pending", merged)
            self.assertIn("Latency regression in service", merged)

    def test_discover_context_files_recent_first_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "old.log"
            p2 = Path(tmp) / "newer.md"
            p3 = Path(tmp) / "newest.txt"
            p4 = Path(tmp) / "ignore.json"
            p1.write_text("old", encoding="utf-8")
            p2.write_text("newer", encoding="utf-8")
            p3.write_text("newest", encoding="utf-8")
            p4.write_text("json", encoding="utf-8")

            now = 1_700_000_000
            os.utime(p1, (now - 30, now - 30))
            os.utime(p2, (now - 20, now - 20))
            os.utime(p3, (now - 10, now - 10))
            os.utime(p4, (now - 5, now - 5))

            picked = discover_context_files(
                from_dir=tmp,
                patterns=["*.log", "*.md", "*.txt"],
                max_files=2,
                recursive=False,
            )
            self.assertEqual(len(picked), 2)
            self.assertTrue(picked[0].endswith("newest.txt"))
            self.assertTrue(picked[1].endswith("newer.md"))

    def test_resolve_context_from_dir_recursive(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "nested"
            sub.mkdir()
            fp = sub / "debug.log"
            fp.write_text("DB timeout seen in trace", encoding="utf-8")

            merged = resolve_context(
                inline_context="",
                from_dir=tmp,
                dir_patterns=["*.log"],
                dir_max_files=5,
                dir_recursive=True,
            )
            self.assertIn("DB timeout seen in trace", merged)

    def test_discover_context_files_respects_max_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "small.log"
            large = Path(tmp) / "large.log"
            small.write_text("ok", encoding="utf-8")
            large.write_text("x" * 50, encoding="utf-8")

            picked = discover_context_files(
                from_dir=tmp,
                patterns=["*.log"],
                max_files=10,
                recursive=False,
                max_file_bytes=10,
            )
            self.assertEqual(len(picked), 1)
            self.assertTrue(picked[0].endswith("small.log"))

    def test_discover_context_files_respects_ignore_patterns_and_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            keep = Path(tmp) / "keep.log"
            skip_by_name = Path(tmp) / "skip-me.log"
            blocked_dir = Path(tmp) / "node_modules"
            blocked_dir.mkdir()
            skip_by_dir = blocked_dir / "module.log"

            keep.write_text("keep", encoding="utf-8")
            skip_by_name.write_text("skip", encoding="utf-8")
            skip_by_dir.write_text("skip", encoding="utf-8")

            picked = discover_context_files(
                from_dir=tmp,
                patterns=["*.log"],
                max_files=10,
                recursive=True,
                ignore_patterns=["skip-*"],
                ignore_dirs=["node_modules"],
            )
            self.assertEqual(len(picked), 1)
            self.assertTrue(picked[0].endswith("keep.log"))

    def test_discover_context_files_with_metadata_reports_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.log"
            p2 = Path(tmp) / "b.log"
            p1.write_text("a", encoding="utf-8")
            p2.write_text("b", encoding="utf-8")
            now = 1_700_000_000
            os.utime(p1, (now - 20, now - 20))
            os.utime(p2, (now - 10, now - 10))

            included, skipped = discover_context_files_with_metadata(
                from_dir=tmp,
                patterns=["*.log"],
                max_files=1,
                recursive=False,
            )
            self.assertEqual(len(included), 1)
            self.assertTrue(any(item["reason"] == "over_max_files" for item in skipped))

    def test_resolve_context_with_metadata_reports_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            keep = Path(tmp) / "keep.log"
            empty = Path(tmp) / "empty.log"
            keep.write_text("timeout in db query", encoding="utf-8")
            empty.write_text("", encoding="utf-8")
            missing = Path(tmp) / "missing.log"

            context, scan = resolve_context_with_metadata(
                inline_context="",
                from_files=[str(keep), str(empty), str(missing)],
                from_git=False,
            )

            self.assertIn("timeout in db query", context)
            self.assertIn(str(keep), scan["included_files"])
            reasons = {item["reason"] for item in scan["skipped_files"]}
            self.assertIn("empty_file", reasons)
            self.assertIn("missing_or_not_file", reasons)
            self.assertIn("skip_summary", scan)
            self.assertIn("by_severity", scan["skip_summary"])
            self.assertIn("remediation_hints", scan)
            self.assertTrue(scan["remediation_hints"])

    def test_skip_severity_prioritization(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "does-not-exist"
            included, skipped = discover_context_files_with_metadata(
                from_dir=str(missing_dir),
                patterns=["*.log"],
            )
            self.assertFalse(included)
            self.assertEqual(skipped[0]["reason"], "missing_or_not_directory")
            self.assertEqual(skipped[0]["severity"], "high")
            self.assertIn("hint", skipped[0])

    def test_new_skip_reason_has_hint(self):
        scan = {
            "included_files": [],
            "skipped_files": [
                {
                    "path": "/tmp/x",
                    "reason": "outside_from_dir",
                    "severity": "low",
                    "severity_score": 1,
                    "hint": "Adjust --from-dir to cover these files if needed.",
                }
            ],
            "skip_summary": {
                "by_reason": {"outside_from_dir": 1},
                "by_severity": {"high": 0, "medium": 0, "low": 1},
            },
            "remediation_hints": [
                {
                    "reason": "outside_from_dir",
                    "severity": "low",
                    "hint": "Adjust --from-dir to cover these files if needed.",
                    "count": 1,
                }
            ],
            "from_git": {"requested": True, "included": True, "reason": ""},
            "git_focus_files": 0,
        }
        text = format_scan_metadata(scan)
        self.assertIn("outside_from_dir", text)

    def test_format_scan_metadata_includes_severity_counts(self):
        scan = {
            "included_files": ["a.log"],
            "skipped_files": [
                {
                    "path": "b.log",
                    "reason": "over_max_bytes",
                    "severity": "medium",
                    "severity_score": 2,
                    "hint": "Increase --dir-max-bytes or narrow file patterns.",
                },
                {
                    "path": "c.log",
                    "reason": "read_error",
                    "severity": "high",
                    "severity_score": 3,
                    "hint": "Verify file permissions and file encoding before rerunning.",
                },
            ],
            "skip_summary": {
                "by_reason": {"over_max_bytes": 1, "read_error": 1},
                "by_severity": {"high": 1, "medium": 1, "low": 0},
            },
            "remediation_hints": [
                {
                    "reason": "read_error",
                    "severity": "high",
                    "hint": "Verify file permissions and file encoding before rerunning.",
                    "count": 1,
                },
                {
                    "reason": "over_max_bytes",
                    "severity": "medium",
                    "hint": "Increase --dir-max-bytes or narrow file patterns.",
                    "count": 1,
                },
            ],
            "from_git": {"requested": False, "included": False, "reason": ""},
        }
        formatted = format_scan_metadata(scan)
        self.assertIn("high=1, medium=1, low=0", formatted)
        self.assertIn("severity=high", formatted)
        self.assertIn("Recommended Remediations:", formatted)
        self.assertIn("Verify file permissions", formatted)
        self.assertLess(formatted.index("c.log"), formatted.index("b.log"))

    def test_custom_scoring_config_changes_ranking(self):
        custom = {
            "action_base_scores": {
                "Temporarily switch to a parallel subtask while preserving context": 2.0
            }
        }
        predictor = DigitalTwinPredictor(scoring_config=custom)
        report = predictor.predict("generic context with no strong signals")
        self.assertEqual(
            report.likely_next_actions[0].title,
            "Temporarily switch to a parallel subtask while preserving context",
        )

    def test_load_scoring_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.json"
            path.write_text(
                '{\"signal_patterns\": {\"benchmark\": \"fps|frametime\"}}',
                encoding="utf-8",
            )
            loaded = load_scoring_config_file(str(path))
            self.assertIn("action_base_scores", loaded)
            self.assertEqual(loaded["signal_patterns"]["benchmark"], "fps|frametime")

    def test_build_snapshot_document_contains_context_hash(self):
        context = "regression in db query latency"
        report = self.p.predict(context)
        payload = build_prediction_payload(report=report, include_scan=False, scan={})
        snapshot = build_snapshot_document(
            prediction_payload=payload,
            context=context,
            snapshot_tag="nightly",
            weights_file="config/weights.local.json",
            show_scan=False,
        )
        self.assertEqual(snapshot["tag"], "nightly")
        self.assertEqual(
            snapshot["context_sha256"], hashlib.sha256(context.encode("utf-8")).hexdigest()
        )
        self.assertEqual(snapshot["context_length_chars"], len(context))
        self.assertIn("prediction", snapshot)
        self.assertEqual(
            snapshot["run_config"]["weights_file"], "config/weights.local.json"
        )

    def test_write_snapshot_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "snapshots" / "one.json"
            snapshot = {
                "schema_version": 1,
                "generated_at_utc": "2026-01-01T00:00:00Z",
                "tag": "test",
                "context_sha256": "abc",
                "context_length_chars": 3,
                "run_config": {"weights_file": "", "show_scan": False},
                "prediction": {"current_intent_inference": "x"},
            }
            written = write_snapshot_file(str(output), snapshot)
            self.assertEqual(written, str(output))
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["tag"], "test")
            self.assertEqual(loaded["prediction"]["current_intent_inference"], "x")

    def test_resolve_project_path_relative_and_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            rel = resolve_project_path(tmp, "nested/file.log")
            self.assertTrue(rel.endswith("nested/file.log"))

            absolute_path = str((Path(tmp) / "abs.txt").resolve())
            absolute = resolve_project_path(tmp, absolute_path)
            self.assertEqual(absolute, absolute_path)

    def test_gather_git_context_for_non_repo_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = gather_git_context(max_commits=3, repo_dir=tmp)
            self.assertEqual(out, "")

    def test_gather_git_changed_files_for_non_repo_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = gather_git_changed_files(max_commits=5, repo_dir=tmp)
            self.assertEqual(files, [])

    def test_founder_audit_detects_missing_readme_pitch_and_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "app/src/main/java/com/example/app/ui").mkdir(parents=True)
            (base / "app/src/main/java/com/example/app/data").mkdir(parents=True)
            (base / "README.md").write_text(
                "# App\n\nA short description only.",
                encoding="utf-8",
            )

            audit = build_founder_audit_report(
                project_dir=tmp,
                context="AndroidManifest.xml build.gradle.kts Retrofit",
                scan={"included_files": []},
                ignore_dirs=[".git", "build"],
                max_files=500,
            )

            first = audit.sections[0]
            build_run = audit.sections[2]
            self.assertTrue(any("Problem -> Solution -> Features -> Demo" in x for x in first.recommendations))
            self.assertTrue(any("No CI workflow detected" in x for x in build_run.recommendations))

    def test_founder_audit_markdown_has_seven_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "README.md").write_text(
                "# Project\n\n## Features\n- x\n## Demo\nlink",
                encoding="utf-8",
            )
            audit = build_founder_audit_report(
                project_dir=tmp,
                context="",
                scan={"included_files": []},
                ignore_dirs=[".git"],
                max_files=500,
            )
            text = audit.to_markdown()
            self.assertIn("1. First impression (10 seconds)", text)
            self.assertIn("7. Product-level thinking (founder mode)", text)

    def test_prediction_payload_includes_founder_audit_when_requested(self):
        report = self.p.predict("simple context")
        audit = build_founder_audit_report(
            project_dir=".",
            context="",
            scan={"included_files": []},
            ignore_dirs=[".git", "build", "dist", "__pycache__", "node_modules", "venv", ".venv", "target"],
            max_files=2000,
        )
        payload = build_prediction_payload(
            report=report,
            include_scan=False,
            scan={},
            founder_audit=audit,
        )
        self.assertIn("founder_audit", payload)
        self.assertEqual(len(payload["founder_audit"]["sections"]), 7)


if __name__ == "__main__":
    unittest.main()
