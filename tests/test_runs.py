from dataclasses import replace

import pytest

from src.config import ExperimentConfig
from src.runs import start_run


def project(tmp_path):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "config.py").write_text("# static source fixture\n")
    (root / "requirements.txt").write_text("# fixture\n")
    return root


def test_numbering_and_resume_preserve_existing(tmp_path):
    root = project(tmp_path)
    first = start_run(root, ExperimentConfig())
    second = start_run(root, ExperimentConfig())
    assert first.path.name == "run_001"
    assert second.path.name == "run_002"
    assert start_run(root, ExperimentConfig(), "resume", "run_001").identity == first.identity


def test_source_and_config_change_rejected(tmp_path):
    root = project(tmp_path)
    run = start_run(root, ExperimentConfig())
    changed = replace(run.config, atol=1e-3)
    with pytest.raises(RuntimeError, match="identity"):
        start_run(root, changed, "resume", run.path.name)
    (root / "src" / "config.py").write_text("# changed\n")
    with pytest.raises(RuntimeError, match="identity"):
        run.require_identity()


def test_stage_records_interrupt_and_releases_lock(tmp_path):
    run = start_run(project(tmp_path), ExperimentConfig())
    with pytest.raises(KeyboardInterrupt):
        with run.stage("train_mha"):
            raise KeyboardInterrupt()
    assert run.read_status()["stages"]["train_mha"]["status"] == "interrupted"
    assert not (run.path / ".stage.lock").exists()


def test_concurrent_stage_and_rebinding_fail(tmp_path):
    run = start_run(project(tmp_path), ExperimentConfig())
    with run.stage("outer"):
        with pytest.raises(RuntimeError, match="owns"):
            with run.stage("inner"):
                pass
    run.bind_data("tokens", {"hash": "first"})
    with pytest.raises(RuntimeError, match="binding"):
        run.bind_data("tokens", {"hash": "changed"})
