"""
Append structured evaluation rows to JSONL and export a combined CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


RESULT_FIELDS: List[str] = [
    "config_hash",
    "paradigm",
    "model_name",
    "conditioning_level",
    "seed",
    "rollout_mse_mean",
    "rollout_mse_final",
    "spectrum_mse",
    "density_mse",
    "infer_ms_per_step_mean",
    "infer_ms_per_step_std",
    "train_gpu_hours",
    "param_count",
]


def append_row(run_dir: Path | str, row: Dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "results.jsonl"
    # Ensure all keys exist for downstream CSV
    out = {k: row.get(k) for k in RESULT_FIELDS}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, default=str) + "\n")


def export_csv(
    results_dir: Path | str,
    out_name: str = "results.csv",
    jsonl_glob: str = "**/results.jsonl",
) -> Path:
    """
    Aggregate every ``results.jsonl`` under ``results_dir`` into a single CSV.
    """
    results_dir = Path(results_dir)
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob(jsonl_glob)):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    out_path = results_dir / out_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in RESULT_FIELDS})
    return out_path


def load_train_results_json(path: Path | str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
