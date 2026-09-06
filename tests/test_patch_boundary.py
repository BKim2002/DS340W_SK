from hashlib import sha256
import json
from pathlib import Path

from ds340w_repro.run_opendelta import prepare_execution_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "vendor" / "OpenDelta"
SERIES = PROJECT_ROOT / "patches" / "series.json"
ALLOWED_CLASSIFICATIONS = {
    "compatibility",
    "configuration",
    "data-handling",
    "provenance",
    "observation-only",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_patch_series_is_hashed_classified_and_narrow() -> None:
    manifest = json.loads(SERIES.read_text(encoding="utf-8"))

    assert manifest["upstream_commit"] == (
        "20e426f716ea53618657cd34bbfc4b347b98cd74"
    )
    assert manifest["patches"]

    for entry in manifest["patches"]:
        patch_path = PROJECT_ROOT / entry["path"]
        assert entry["classification"] in ALLOWED_CLASSIFICATIONS
        assert _digest(patch_path) == entry["sha256"]
        assert entry["touched_paths"]
        assert all(
            path.startswith("examples/legacies/examples_seq2seq/")
            for path in entry["touched_paths"]
        )


def test_patches_apply_to_a_disposable_tree_only(tmp_path: Path) -> None:
    original_runner = UPSTREAM / "examples/legacies/examples_seq2seq/run_seq2seq.py"
    before = _digest(original_runner)

    report = prepare_execution_tree(PROJECT_ROOT, tmp_path / "runtime")

    assert _digest(original_runner) == before
    assert report.upstream_commit == (
        "20e426f716ea53618657cd34bbfc4b347b98cd74"
    )
    assert report.pre_patch_runner_sha256 == before
    assert report.post_patch_runner_sha256 != before
    assert report.runner == (
        tmp_path
        / "runtime"
        / "examples/legacies/examples_seq2seq/run_seq2seq.py"
    )

    patched_runner = report.runner.read_text(encoding="utf-8")
    patched_tasks = (
        report.root
        / "examples/legacies/examples_seq2seq/data_processors/tasks.py"
    ).read_text(encoding="utf-8")
    assert "DS340W_SST2_TRAIN_FILE" in patched_tasks
    assert "DS340W_SST2_TEST_FILE" in patched_tasks
    assert "datasets.load_dataset('json'" in patched_tasks
    assert "SST2 course-data files were not provided" in patched_tasks
    assert "if training_args.do_test:" in patched_runner
    assert 'data_info["test"]' in patched_runner
    assert "repo_name = create_hub_repo_name(" not in patched_runner


def test_patches_do_not_add_a_replacement_training_loop() -> None:
    forbidden_additions = (
        "optimizer.step(",
        ".backward(",
        "for batch in",
        "AdamW(",
        "LoraModel(",
    )

    for entry in json.loads(SERIES.read_text(encoding="utf-8"))["patches"]:
        text = (PROJECT_ROOT / entry["path"]).read_text(encoding="utf-8")
        added_lines = "\n".join(
            line[1:]
            for line in text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        assert not any(token in added_lines for token in forbidden_additions)
