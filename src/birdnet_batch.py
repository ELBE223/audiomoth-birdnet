# Reusable BirdNET batch utilities (CSV in/out)
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import csv

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
CSV_HEADER = ["file", "start_s", "end_s", "label", "confidence"]

def find_audio_files(base: Path, pattern: str | None) -> List[Path]:
    """Find audio files under base. Pattern filters folders (glob)."""
    base = Path(base).expanduser().resolve()
    if not base.exists():
        return []
    roots = [d for d in base.rglob(pattern) if d.is_dir()] if pattern else []
    if not roots:
        roots = [base]
    files: List[Path] = []
    for r in roots:
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
    seen, out = set(), []
    for f in files:
        if f not in seen:
            out.append(f); seen.add(f)
    return out

def analyze_one_to_csv(file_path: Path, out_dir: Path, min_conf: float = 0.1) -> Path:
    """Analyze one file and write a CSV into results/<parent>/<file>.csv"""
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer

    file_path = Path(file_path)
    out_dir = Path(out_dir).expanduser().resolve()
    sub = out_dir / file_path.parent.name
    sub.mkdir(parents=True, exist_ok=True)
    out_csv = sub / f"{file_path.stem}.csv"

    analyzer = Analyzer()
    rec = Recording(analyzer, str(file_path), min_conf=min_conf)
    rec.analyze()

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for d in (rec.detections or []):
            w.writerow([
                file_path.name,
                d.get("start_time"),
                d.get("end_time"),
                d.get("common_name"),
                f"{float(d.get('confidence', 0.0)):.3f}",
            ])
    return out_csv

def analyze_batch_to_csv(files: Iterable[Path], out_dir: Path,
                         min_conf: float = 0.1, workers: int = 1) -> list[Path]:
    """Analyze many files in parallel; write per-file CSVs."""
    from multiprocessing import Pool

    files = list(files)
    args = [(Path(f), Path(out_dir), float(min_conf)) for f in files]

    def _worker(a):
        f, o, t = a
        return analyze_one_to_csv(f, o, t)

    if workers <= 1:
        return [_worker(a) for a in args]
    with Pool(processes=workers) as pool:
        return list(pool.imap_unordered(_worker, args))

def compile_master_csv(out_dir: Path, master_name: str = "master_results.csv") -> Path:
    """Merge all per-file CSVs into one master CSV with header."""
    out_dir = Path(out_dir).expanduser().resolve()
    csvs = [p for p in out_dir.rglob("*.csv") if p.name != master_name]
    master = out_dir / master_name
    with master.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for p in csvs:
            with p.open("r", encoding="utf-8") as r:
                reader = csv.reader(r)
                try:
                    first = next(reader)
                except StopIteration:
                    continue
                # write first row if not header
                if [x.strip().lower() for x in first] != [x.lower() for x in CSV_HEADER]:
                    w.writerow(first)
                for row in reader:
                    if row:
                        w.writerow(row)
    return master
