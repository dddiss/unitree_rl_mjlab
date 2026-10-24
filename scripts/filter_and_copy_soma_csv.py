"""Filter and copy motion CSVs using SONIC's Bones filename keyword rules.

This matches keywords against the immediate parent directory plus the filename,
case-insensitively, without inspecting motion contents or physical feasibility.
It works on both original SOMA/Bones CSVs and converted LAFAN CSVs. Files at the
source root and all nested levels are included; the entire relative path is
preserved, so equal filenames in different subdirectories stay separate.

Example:
  python scripts/filter_and_copy_soma_csv.py --source /data/bones_lafan \
    --dest /data/bones_lafan_filtered --workers 16

--dry-run previews without writing; --filter-keywords replaces the default list
(an empty list disables exclusions); --add-keywords extends it. --filter-file
optionally supplies one inclusion substring per line; exclusions take priority.
Existing files are skipped unless --overwrite is given. Use a fresh destination
when changing filters: this script does not remove previously copied files.
"""

import argparse
from multiprocessing.pool import ThreadPool
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile


# Keep SONIC filter_and_copy_bones_data.py's default keyword list unchanged.
DEFAULT_FILTER_KEYWORDS = (
  "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on",
  "seat", "table", "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_",
  "scooter", "stepdown", "acrobatics_", "box_HSPU", "cartwheel", "50cm_box_",
  "on_box", "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m",
  "230m", "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll",
  "kozak_dance", "stair", "handstand", "box_jump", "monkey_jump", "safety_roll",
  "box_dips", "walking_on_edge", "push_obstacle",
)


def should_filter_out(
  name: str,
  filter_keywords: tuple[str, ...] | list[str] = DEFAULT_FILTER_KEYWORDS,
  include_keywords: list[str] | None = None,
) -> bool:
  name = name.lower()
  return any(word.lower() in name for word in filter_keywords) or (
    include_keywords is not None and not any(word.lower() in name for word in include_keywords)
  )


def _copy_one(job: tuple[Path, Path, bool, bool]) -> tuple[str, str, str]:
  source, output, overwrite, dry_run = job
  temporary = None
  try:
    if output.is_file() and not overwrite:
      return str(source), "skipped", ""
    if not dry_run:
      output.parent.mkdir(parents=True, exist_ok=True)
      with NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
      shutil.copy2(source, temporary)
      temporary.replace(output)
    return str(source), "copied", ""
  except Exception as error:
    return str(source), "failed", str(error)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def filter_directory(
  source: Path,
  dest: Path,
  filter_keywords: tuple[str, ...] | list[str] = DEFAULT_FILTER_KEYWORDS,
  add_keywords: list[str] | None = None,
  filter_file: Path | None = None,
  workers: int = 8,
  dry_run: bool = False,
  verbose: bool = False,
  overwrite: bool = False,
) -> int:
  source, dest = source.expanduser().resolve(), dest.expanduser().resolve()
  if not source.is_dir():
    raise ValueError(f"Source directory does not exist: {source}")
  if source == dest or source in dest.parents or dest in source.parents:
    raise ValueError("Source and destination must be separate, non-nested directories")
  if workers < 1:
    raise ValueError("--workers must be positive")
  keywords = [*filter_keywords, *(add_keywords or [])]
  include = None
  if filter_file is not None:
    include = [line.strip() for line in filter_file.expanduser().read_text().splitlines() if line.strip()]
  files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
  if not files:
    raise ValueError(f"No CSV files found in {source}")
  jobs = []
  filtered = 0
  for path in files:
    relative = path.relative_to(source)
    if should_filter_out(f"{path.parent.name}/{path.name}", keywords, include):
      filtered += 1
      if verbose:
        print(f"[FILTERED] {relative}")
    else:
      jobs.append((path, dest / relative, overwrite, dry_run))
  counts = {"copied": 0, "skipped": 0, "failed": 0}

  def consume(results):
    for path, status, error in results:
      counts[status] += 1
      if error:
        print(f"[FAILED] {path}: {error}", flush=True)
      elif verbose:
        label = "WOULD COPY" if dry_run and status == "copied" else status.upper()
        print(f"[{label}] {Path(path).relative_to(source)}")

  if workers == 1 or len(jobs) < 2:
    consume(map(_copy_one, jobs))
  else:
    with ThreadPool(min(workers, len(jobs))) as pool:
      consume(pool.imap_unordered(_copy_one, jobs, chunksize=16))
  action = "would copy" if dry_run else "copied"
  print(f"Finished: {len(files)} found, {counts['copied']} {action}, {filtered} filtered, "
        f"{counts['skipped']} skipped, {counts['failed']} failed.")
  print(f"CSV directory: {dest}")
  if dry_run:
    print("Dry run: no files written.")
  return int(counts["failed"] > 0)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--source", required=True, type=Path)
  parser.add_argument("--dest", required=True, type=Path)
  parser.add_argument("--filter-keywords", nargs="*", default=DEFAULT_FILTER_KEYWORDS)
  parser.add_argument("--add-keywords", nargs="+")
  parser.add_argument("--filter-file", "--filter_file", type=Path)
  parser.add_argument("--workers", type=int, default=8, help="Parallel file-copy workers")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--verbose", "-v", action="store_true")
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args(argv)
  try:
    return filter_directory(**vars(args))
  except (ValueError, OSError) as error:
    parser.error(str(error))


if __name__ == "__main__":
  raise SystemExit(main())
