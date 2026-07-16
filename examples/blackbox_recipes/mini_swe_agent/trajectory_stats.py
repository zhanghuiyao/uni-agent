"""Summarize scores from one mini-swe-agent inference run.

Usage:
    python examples/blackbox_recipes/mini_swe_agent/trajectory_stats.py RUN_DIR
    python examples/blackbox_recipes/mini_swe_agent/trajectory_stats.py RUN_DIR \
        --json-output stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

_ARTIFACT_SCHEMA_VERSION = 1
_TRAJECTORY_GLOB = "trajectory_session_*_task_*.jsonl"
_TRAJECTORY_FILENAME = re.compile(r"trajectory_session_(\d+)_task_(\d+)\.jsonl")
_REQUIRED_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "dataset_index",
        "uid",
        "session_index",
        "trajectory_count",
        "final_trajectory_index",
        "score",
        "resolved",
    }
)


class TrajectoryStatsError(ValueError):
    """Raised when trajectory artifacts cannot be summarized safely."""


@dataclass(frozen=True)
class SessionScore:
    dataset_index: int
    session_index: int
    score: float
    resolved: bool
    file_name: str


def _require_int(record: dict[str, Any], key: str, *, minimum: int) -> int:
    value = record[key]
    if type(value) is not int or value < minimum:
        raise TrajectoryStatsError(f"metadata field {key!r} must be an integer >= {minimum}")
    return value


def _read_session_score(path: Path) -> SessionScore:
    match = _TRAJECTORY_FILENAME.fullmatch(path.name)
    if match is None:
        raise TrajectoryStatsError(f"{path}: invalid trajectory filename")

    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    if not first_line:
        raise TrajectoryStatsError(f"{path}: empty trajectory file")

    try:
        record = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise TrajectoryStatsError(f"{path}: invalid JSON in session metadata: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise TrajectoryStatsError(f"{path}: session metadata must be a JSON object")

    missing = sorted(_REQUIRED_METADATA_FIELDS - record.keys())
    if missing:
        raise TrajectoryStatsError(f"{path}: session metadata is missing fields: {', '.join(missing)}")
    schema_version = record["schema_version"]
    if type(schema_version) is not int or schema_version != _ARTIFACT_SCHEMA_VERSION:
        raise TrajectoryStatsError(
            f"{path}: unsupported schema_version {schema_version!r}; expected {_ARTIFACT_SCHEMA_VERSION}"
        )
    if record["record_type"] != "session_metadata":
        raise TrajectoryStatsError(f"{path}: first record must have record_type='session_metadata'")
    if not isinstance(record["uid"], str) or not record["uid"]:
        raise TrajectoryStatsError(f"{path}: metadata field 'uid' must be a non-empty string")

    dataset_index = _require_int(record, "dataset_index", minimum=0)
    session_index = _require_int(record, "session_index", minimum=0)
    trajectory_count = _require_int(record, "trajectory_count", minimum=1)
    final_trajectory_index = _require_int(record, "final_trajectory_index", minimum=0)
    if final_trajectory_index != trajectory_count - 1:
        raise TrajectoryStatsError(
            f"{path}: final_trajectory_index must equal trajectory_count - 1 "
            f"({trajectory_count - 1})"
        )

    score_value = record["score"]
    if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
        raise TrajectoryStatsError(f"{path}: metadata field 'score' must be numeric")
    score = float(score_value)
    if not math.isfinite(score):
        raise TrajectoryStatsError(f"{path}: metadata field 'score' must be finite")

    resolved = record["resolved"]
    if not isinstance(resolved, bool):
        raise TrajectoryStatsError(f"{path}: metadata field 'resolved' must be boolean")
    if resolved != (score > 0.0):
        raise TrajectoryStatsError(f"{path}: metadata field 'resolved' must equal (score > 0)")

    filename_session_index = int(match.group(1))
    filename_dataset_index = int(match.group(2))
    if (filename_dataset_index, filename_session_index) != (dataset_index, session_index):
        raise TrajectoryStatsError(
            f"{path}: filename task/session ({filename_dataset_index}, {filename_session_index}) "
            f"does not match metadata ({dataset_index}, {session_index})"
        )

    return SessionScore(
        dataset_index=dataset_index,
        session_index=session_index,
        score=score,
        resolved=resolved,
        file_name=path.name,
    )


def load_session_scores(run_dir: Path) -> tuple[Path, list[SessionScore]]:
    run_dir = run_dir.expanduser()
    if not run_dir.exists():
        raise TrajectoryStatsError(f"run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise TrajectoryStatsError(f"run path is not a directory: {run_dir}")
    run_dir = run_dir.resolve()

    paths = sorted(path for path in run_dir.glob(_TRAJECTORY_GLOB) if path.is_file())
    if not paths:
        raise TrajectoryStatsError(f"no {_TRAJECTORY_GLOB} files found in {run_dir}")

    sessions: list[SessionScore] = []
    seen: dict[tuple[int, int], str] = {}
    for path in paths:
        session = _read_session_score(path)
        key = (session.dataset_index, session.session_index)
        if key in seen:
            raise TrajectoryStatsError(
                f"duplicate task/session ({session.dataset_index}, {session.session_index}) "
                f"in {seen[key]} and {session.file_name}"
            )
        seen[key] = session.file_name
        sessions.append(session)

    return run_dir, sessions


def build_report(run_dir: Path, sessions: list[SessionScore]) -> dict[str, Any]:
    grouped: dict[int, list[SessionScore]] = defaultdict(list)
    for session in sessions:
        grouped[session.dataset_index].append(session)

    tasks = []
    for dataset_index in sorted(grouped):
        task_sessions = sorted(grouped[dataset_index], key=lambda item: item.session_index)
        mean_score = fmean(session.score for session in task_sessions)
        tasks.append(
            {
                "dataset_index": dataset_index,
                "session_count": len(task_sessions),
                "sessions": [
                    {
                        "session_index": session.session_index,
                        "score": session.score,
                        "resolved": session.resolved,
                        "file": session.file_name,
                    }
                    for session in task_sessions
                ],
                "mean_score": mean_score,
                "resolved": any(session.resolved for session in task_sessions),
            }
        )

    resolved_session_count = sum(session.resolved for session in sessions)
    resolved_task_count = sum(task["resolved"] for task in tasks)
    distribution = Counter(session.score for session in sessions)
    return {
        "run_dir": str(run_dir),
        "session_count": len(sessions),
        "task_count": len(tasks),
        "resolved_session_count": resolved_session_count,
        "session_resolve_rate": resolved_session_count / len(sessions),
        "mean_session_score": fmean(session.score for session in sessions),
        "score_distribution": [
            {"score": score, "count": distribution[score]} for score in sorted(distribution)
        ],
        "resolved_task_count": resolved_task_count,
        "task_resolve_rate": resolved_task_count / len(tasks),
        "mean_task_score": fmean(task["mean_score"] for task in tasks),
        "tasks": tasks,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"Run directory: {report['run_dir']}",
        f"Sessions: {report['session_count']}",
        f"Tasks: {report['task_count']}",
        (
            f"Resolved sessions: {report['resolved_session_count']}/{report['session_count']} "
            f"({report['session_resolve_rate']:.2%})"
        ),
        f"Mean session score: {report['mean_session_score']:.6f}",
        "Score distribution:",
    ]
    lines.extend(f"  {item['score']!r}: {item['count']}" for item in report["score_distribution"])
    lines.extend(
        [
            (
                f"Resolved tasks: {report['resolved_task_count']}/{report['task_count']} "
                f"({report['task_resolve_rate']:.2%})"
            ),
            f"Mean task score: {report['mean_task_score']:.6f}",
            "",
            "Per-task scores:",
        ]
    )
    for task in report["tasks"]:
        scores = ", ".join(f"{item['session_index']}:{item['score']!r}" for item in task["sessions"])
        lines.append(
            f"  Task {task['dataset_index']}: sessions={task['session_count']} scores=[{scores}] "
            f"mean={task['mean_score']:.6f} resolved={'yes' if task['resolved'] else 'no'}"
        )
    return "\n".join(lines)


def _write_json_output(path: Path, report: dict[str, Any]) -> Path:
    path = path.expanduser()
    if not path.parent.is_dir():
        raise TrajectoryStatsError(f"JSON output parent directory does not exist: {path.parent}")
    if path.is_dir():
        raise TrajectoryStatsError(f"JSON output path is a directory: {path}")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize mini-swe-agent trajectory scores")
    parser.add_argument("run_dir", type=Path, help="One timestamped inference run directory")
    parser.add_argument("--json-output", type=Path, help="Also write the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_dir, sessions = load_session_scores(args.run_dir)
        report = build_report(run_dir, sessions)
        json_output = _write_json_output(args.json_output, report) if args.json_output else None
    except (OSError, TrajectoryStatsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    if json_output is not None:
        print(f"JSON report: {json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
