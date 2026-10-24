"""Batch routing tests; simulation/data conversion stays in csv_to_npz.py."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/csv_to_npz_multi.py"
spec = importlib.util.spec_from_file_location("csv_to_npz_multi", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@pytest.fixture
def converter(tmp_path, monkeypatch):
  """Run real subprocesses with a lightweight converter to test CLI forwarding."""
  script = tmp_path / "fake_converter.py"
  script.write_text('''
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
for name in ("robot", "input-file", "input-fps", "output-fps", "output-dir", "device", "save-video"):
    p.add_argument("--" + name, required=True)
p.add_argument("--video-dir")
p.add_argument("--line-range", nargs=2, type=int)
a = p.parse_args()
source = Path(a.input_file)
with open("calls.jsonl", "a") as calls:
    calls.write(json.dumps(vars(a)) + "\\n")
if source.stem == "broken":
    raise SystemExit(1)
if source.stem == "missing_output":
    raise SystemExit(0)
out = Path(a.output_dir) / (source.stem + ".npz")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("converted")
if a.save_video == "True":
    video = Path(a.video_dir) / (source.stem + ".mp4")
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_text("video")
''')
  monkeypatch.setattr(batch, "CONVERTER", script)
  monkeypatch.setattr(batch, "PROJECT_ROOT", tmp_path)
  return tmp_path


def test_recursive_paths_and_parameters(converter):
  import json

  source = converter / "input motions"
  for name in ("a/dance.csv", "b/dance.CSV", "root.csv", "ignored.txt"):
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("input")
  output, video = converter / "npz", converter / "videos"
  assert batch.main([
    "--robot", "g1", "--input-dir", str(source), "--output-dir", str(output),
    "--recursive", "--input-fps", "60", "--output-fps", "50", "--device", "cpu",
    "--save-video", "True", "--video-dir", str(video), "--line-range", "2", "20",
  ]) == 0
  assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*.npz")) == [
    "a/dance.npz", "b/dance.npz", "root.npz"]
  assert sorted(p.relative_to(video).as_posix() for p in video.rglob("*.mp4")) == [
    "a/dance.mp4", "b/dance.mp4", "root.mp4"]
  calls = [json.loads(line) for line in (converter / "calls.jsonl").read_text().splitlines()]
  assert len(calls) == 3
  for call in calls:
    assert call["robot"] == "g1" and call["device"] == "cpu"
    assert float(call["input_fps"]) == 60 and float(call["output_fps"]) == 50
    assert call["line_range"] == [2, 20]


def test_default_directory_nonrecursive_and_video_disabled(converter):
  source = converter / "input"
  (source / "nested").mkdir(parents=True)
  (source / "dance.csv").touch()
  (source / "nested/other.csv").touch()
  assert batch.main(["--robot", "g1_23dof", "--input-dir", str(source),
                     "--save-video", "False"]) == 0
  output = converter / "src/assets/motions/g1_23dof"
  assert [p.name for p in output.iterdir()] == ["dance.npz"]


def test_skip_overwrite_and_missing_video(converter):
  source, output = converter / "input", converter / "output"
  source.mkdir(); output.mkdir()
  (source / "dance.csv").touch()
  npz, video = output / "dance.npz", output / "dance.mp4"
  npz.write_text("existing")
  args = ["--robot", "g1", "--input-dir", str(source), "--output-dir", str(output)]
  assert batch.main(args) == 0
  assert not (converter / "calls.jsonl").exists()
  assert npz.read_text() == "existing"
  # With video requested, a missing MP4 requires regenerating this motion.
  assert batch.main([*args, "--save-video"]) == 0
  assert npz.read_text() == "converted" and video.is_file()
  npz.write_text("keep"); video.write_text("keep")
  assert batch.main([*args, "--save-video"]) == 0
  assert npz.read_text() == video.read_text() == "keep"
  assert batch.main([*args, "--save-video", "--overwrite"]) == 0
  assert npz.read_text() == "converted" and video.read_text() == "video"


def test_failure_continues_and_returns_nonzero(converter, capsys):
  source, output = converter / "input", converter / "output"
  source.mkdir()
  for name in ("broken.csv", "missing_output.csv", "success.csv"):
    (source / name).touch()
  assert batch.main(["--robot", "g1", "--input-dir", str(source), "--output-dir", str(output)]) == 1
  assert (output / "success.npz").is_file()
  assert not (output / "broken.npz").exists()
  text = capsys.readouterr().out
  assert "1 converted, 0 skipped, 2 failed" in text


def test_empty_input_reports_error_without_starting_converter(converter):
  with pytest.raises(SystemExit) as error:
    batch.main(["--robot", "g1", "--input-dir", str(converter)])
  assert error.value.code == 2
  assert not (converter / "calls.jsonl").exists()


def test_help_requires_no_simulation_imports():
  result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
  assert result.returncode == 0
  assert "--recursive" in result.stdout and "--output-dir" in result.stdout
