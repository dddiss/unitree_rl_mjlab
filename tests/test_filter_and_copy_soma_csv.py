"""Filename filtering leaves CSV contents intact and preserves clip identities."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/filter_and_copy_soma_csv.py"
spec = importlib.util.spec_from_file_location("filter_and_copy_soma_csv", SCRIPT)
filtering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filtering)


def write_files(root, names):
  for i, name in enumerate(names):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{i},1,2,3\n")


def test_recursive_copy_keywords_and_identical_names(tmp_path):
  source, dest = tmp_path / "input", tmp_path / "filtered"
  write_files(source, [
    "a/walk.csv", "b/deep/walk.csv", "root.CSV", "session/Chair_turn.csv",
    "BIKE_session/walk.csv", "walk.txt",
  ])
  # Exercise concurrent copying, including root-level and deeply nested CSVs.
  assert filtering.main(["--source", str(source), "--dest", str(dest), "--workers", "2"]) == 0
  expected = ["a/walk.csv", "b/deep/walk.csv", "root.CSV"]
  assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == expected
  for name in expected:
    assert (dest / name).read_bytes() == (source / name).read_bytes()
  assert len(list(source.rglob("*.csv"))) == 4  # Original files are retained.


def test_preview_skip_and_overwrite(tmp_path, capsys):
  source, dest = tmp_path / "input", tmp_path / "filtered"
  write_files(source, ["walk.csv", "handstand.csv"])
  args = ["--source", str(source), "--dest", str(dest), "--workers", "1"]
  assert filtering.main([*args, "--dry-run", "--verbose"]) == 0
  assert not dest.exists()
  assert "1 would copy" in capsys.readouterr().out
  assert filtering.main(args) == 0
  original = (dest / "walk.csv").read_bytes()
  (source / "walk.csv").write_text("new contents")
  assert filtering.main(args) == 0
  assert (dest / "walk.csv").read_bytes() == original
  assert filtering.main([*args, "--overwrite"]) == 0
  assert (dest / "walk.csv").read_text() == "new contents"


def test_custom_exclusions_additions_and_include_file(tmp_path):
  source, dest = tmp_path / "input", tmp_path / "filtered"
  write_files(source, ["walk.csv", "dance.csv", "chair.csv", "bad_walk.csv"])
  include = tmp_path / "include.txt"
  include.write_text("walk\nchair\n")
  assert filtering.main([
    "--source", str(source), "--dest", str(dest), "--filter-keywords", "dance",
    "--add-keywords", "bad", "--filter_file", str(include), "--workers", "1",
  ]) == 0
  assert sorted(p.name for p in dest.iterdir()) == ["chair.csv", "walk.csv"]
  # Explicitly empty exclusions also disable default matches such as 'chair'.
  assert filtering.main([
    "--source", str(source), "--dest", str(tmp_path / "all"), "--filter-keywords",
  ]) == 0
  assert len(list((tmp_path / "all").iterdir())) == 4


def test_copy_failure_is_reported_and_does_not_leave_partial_file(tmp_path, monkeypatch, capsys):
  source, dest = tmp_path / "input", tmp_path / "filtered"
  write_files(source, ["walk.csv", "run.csv"])
  copy2 = filtering.shutil.copy2

  def fail_walk(path, target):
    if path.name == "walk.csv":
      raise OSError("copy failed")
    return copy2(path, target)

  monkeypatch.setattr(filtering.shutil, "copy2", fail_walk)
  assert filtering.filter_directory(source, dest, workers=1) == 1
  assert "copy failed" in capsys.readouterr().out
  assert (dest / "run.csv").exists()
  assert not (dest / "walk.csv").exists()
  assert not list(dest.glob("*.tmp"))


def test_nested_destination_is_rejected(tmp_path):
  write_files(tmp_path / "input", ["walk.csv"])
  with pytest.raises(SystemExit):
    filtering.main(["--source", str(tmp_path / "input"), "--dest", str(tmp_path / "input/out")])


def test_cli_help_needs_no_simulator():
  result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"], capture_output=True, text=True)
  assert result.returncode == 0
  assert "--filter-keywords" in result.stdout and "--dry-run" in result.stdout
