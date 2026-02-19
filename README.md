# Digital Twin Future Work Predictor

Dependency-free CLI that predicts likely next technical actions from project context using deterministic engineering-pattern rules.

## Why this project

This provides a concrete starting point for your behavioral-digital-twin idea:
- Ranked likely actions with probability estimates
- Intent inference for what you're likely trying to build/fix/optimize
- Ranked future improvements inferred from current project trajectory
- Style deviation detection against your baseline tendencies
- Alternative path ranking with explicit triggers

## Run

```bash
python3 src/predictor.py --context "Regression after refactor; throughput dropped; stack traces show DB timeout"
```

Interactive mode:

```bash
python3 src/predictor.py
```

JSON mode:

```bash
python3 src/predictor.py --context "..." --json
```

Use a custom scoring config:

```bash
python3 src/predictor.py --context "fps dropped after refactor" --weights-file config/weights.example.json
```

`weights.example.json` also supports `signal_thresholds` so you can require stronger evidence
before a signal is considered active (useful when scanning large doc/code contexts).

Start from template:

```bash
cp config/weights.example.json config/weights.local.json
python3 src/predictor.py --from-dir . --weights-file config/weights.local.json
```

Use project artifacts as context:

```bash
python3 src/predictor.py --from-files notes.txt debug.log issue.md
```

Run against another project directory (without `cd`):

```bash
python3 src/predictor.py \
  --project-dir /path/to/other-project \
  --from-dir . \
  --from-git \
  --show-scan
```

This mode uses that project's files + commit history to produce both
near-term predictions and ranked future improvements.
Android/Kotlin projects are auto-detected and receive mobile-specific
future improvements (Compose/UI tests, startup baseline profiles, release health).
When `--from-git` is enabled, recently changed files are automatically prioritized.
Tune this with `--git-file-focus-limit` (set `0` to disable git-based file focus).

Run founder-style improvement audit (7 sections) on any project:

```bash
python3 src/predictor.py \
  --project-dir /path/to/other-project \
  --from-dir app/src/main \
  --dir-patterns "*.kt" "*.java" "*.xml" \
  --from-git --git-commits 80 \
  --founder-audit \
  --audit-only
```

`--founder-audit` adds a concrete checklist-style audit covering:
first impression, architecture, build/run, code quality,
security/reliability, performance/scalability, and product thinking.
Use `--audit-only` to print only this audit block.
If the project is large, increase coverage with `--audit-max-files 12000`.

Auto-discover recent context files from a directory:

```bash
python3 src/predictor.py --from-dir . --dir-max-files 12
```

Control discovery patterns and recursion:

```bash
python3 src/predictor.py --from-dir . --dir-patterns "*.log" "*.md" --non-recursive
```

Keep scan high-signal (size cap + ignores):

```bash
python3 src/predictor.py \
  --from-dir . \
  --dir-max-files 12 \
  --dir-max-bytes 200000 \
  --ignore-dirs .git node_modules venv __pycache__ \
  --ignore-patterns "*/archive/*" "skip-*"
```

Context ingestion is source-aware: plain text/log inputs are weighted highest, while source/test
files are downweighted to reduce false positives from keyword-heavy code.

Show ingestion diagnostics (included/skipped files):

```bash
python3 src/predictor.py --from-dir . --show-scan
python3 src/predictor.py --from-dir . --show-scan --json
```

`--show-scan` now includes skip severity counts (`high`, `medium`, `low`) and
prioritizes high-severity skip reasons first.
It also prints recommended remediations for each skip reason.

Export reproducible prediction snapshots:

```bash
python3 src/predictor.py \
  --from-dir . \
  --show-scan \
  --snapshot-out snapshots/run-001.json \
  --snapshot-tag "baseline"
```

Combine manual context + files + recent git history:

```bash
python3 src/predictor.py \
  --context "Current issue: timeout after cache rewrite" \
  --from-files debug.log perf.txt \
  --from-git \
  --git-commits 12
```

## Test

```bash
python3 -m unittest -v tests/test_predictor.py
```

## Next engineering increments

1. Add context adapters (git commits, issue text, benchmark logs).
2. Add snapshot diff utility for cross-run comparison.
3. Add regression corpus and calibration checks for probability stability.
