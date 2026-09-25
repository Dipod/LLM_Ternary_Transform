"""Benchmark evaluation through ``lm-evaluation-harness`` (design.md D4).

The harness is invoked as a subprocess because its task registry and dependency
pins are independent of this package. Task keys, few-shot counts, the item limit
and the harness version are recorded with every result so the reference and the
compressed model are always compared under the same configuration (spec R5).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Metric keys tried in order, per harness convention, when reading a task's score.
_PREFERRED_METRICS: tuple[str, ...] = (
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "acc_norm",
    "acc",
    "exact_match",
)


@dataclass(frozen=True)
class HarnessResult:
    """Parsed benchmark result for one model.

    Attributes:
        version: Harness version string, when the harness reported one.
        tasks: Primary metric value in points per task.
        raw: The full parsed result payload.
        output_path: Path of the JSON file the harness wrote.
    """

    version: str
    tasks: Mapping[str, float]
    raw: Mapping[str, Any]
    output_path: str


def parse_results(raw: Mapping[str, Any]) -> dict[str, float]:
    """Extract one primary metric value per task from a harness payload.

    Args:
        raw: Parsed harness result JSON.

    Returns:
        Task name to metric value in points.

    Raises:
        ValueError: If the payload has no ``results`` mapping or a task exposes
            no numerical metric.
    """
    results = raw.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("harness payload has no 'results' mapping")
    parsed: dict[str, float] = {}
    for task, entry in results.items():
        if not isinstance(entry, Mapping):
            continue
        value: float | None = None
        for key in _PREFERRED_METRICS:
            candidate = entry.get(key)
            if isinstance(candidate, int | float):
                value = float(candidate) * 100.0
                break
        if value is None:
            for key, candidate in entry.items():
                if key == "alias" or not isinstance(candidate, int | float):
                    continue
                value = float(candidate) * 100.0
                break
        if value is None:
            raise ValueError(f"task {task!r} exposes no numerical metric")
        parsed[str(task)] = value
    if not parsed:
        raise ValueError("harness payload contains no tasks")
    return parsed


def _locate_results(output_path: Path) -> Path:
    """Return the harness's result file for a requested output path.

    ``lm-evaluation-harness`` 0.4.13 inserts a timestamp before the extension
    (``harness_2026-09-25T01-27-51.json``) instead of writing the requested name,
    so the newest matching sibling is used.

    Args:
        output_path: The path passed to ``--output_path``.

    Returns:
        The result file to parse.

    Raises:
        FileNotFoundError: If no result file exists for the requested path.
    """
    if output_path.is_file():
        return output_path
    candidates = sorted(
        output_path.parent.glob(f"{output_path.stem}_*{output_path.suffix}"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"harness wrote no result file for {output_path} "
            f"(looked for {output_path.stem}_*{output_path.suffix})"
        )
    return candidates[-1]


def build_command(
    model_id: str,
    *,
    tasks: Sequence[str],
    num_fewshot: Sequence[int],
    limit: int | None,
    batch_size: int,
    device: str,
    output_path: Path,
    python_executable: str,
) -> list[str]:
    """Build the harness command line.

    Args:
        model_id: HuggingFace model id or local path.
        tasks: Harness task names.
        num_fewshot: Few-shot counts aligned with ``tasks``.
        limit: Item cap per task, or ``None`` for the full task.
        batch_size: Harness batch size.
        device: Torch device string passed to the harness.
        output_path: Where the harness writes its JSON result.
        python_executable: Interpreter that has ``lm_eval`` installed.

    Returns:
        The argv list.

    Raises:
        ValueError: If ``tasks`` and ``num_fewshot`` lengths differ.
    """
    if len(tasks) != len(num_fewshot):
        raise ValueError(f"tasks and num_fewshot must align: {len(tasks)} vs {len(num_fewshot)}")
    command = [
        python_executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_id},dtype=bfloat16",
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        ",".join(str(n) for n in num_fewshot),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--output_path",
        str(output_path),
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def run_harness(
    model_id: str,
    *,
    tasks: Sequence[str],
    num_fewshot: Sequence[int],
    limit: int | None,
    batch_size: int,
    device: str,
    output_dir: Path,
    python_executable: str | None = None,
) -> HarnessResult:
    """Run the harness and return the parsed result.

    Args:
        model_id: HuggingFace model id or local path.
        tasks: Harness task names.
        num_fewshot: Few-shot counts aligned with ``tasks``.
        limit: Item cap per task, or ``None``.
        batch_size: Harness batch size.
        device: Torch device string.
        output_dir: Directory to write the harness JSON into.
        python_executable: Interpreter with ``lm_eval``; defaults to the running one.

    Returns:
        The parsed :class:`HarnessResult`.

    Raises:
        RuntimeError: If the harness exits non-zero; the message carries the tail
            of its combined output.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "harness.json"
    executable = python_executable or sys.executable
    command = build_command(
        model_id,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        device=device,
        output_path=output_path,
        python_executable=executable,
    )
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr)[-4000:]
        raise RuntimeError(f"lm_eval failed with exit code {completed.returncode}:\n{tail}")
    raw: Any = json.loads(_locate_results(output_path).read_text(encoding="utf-8"))
    version = str(raw.get("lm_eval_version", raw.get("version", "unknown")))
    return HarnessResult(
        version=version,
        tasks=parse_results(raw),
        raw=raw,
        output_path=str(_locate_results(output_path)),
    )
