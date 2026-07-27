"""Unit tests for core.provenance module (provenance manifest generation and export)."""

import json
from pathlib import Path

import pytest

from core.config import ProcessingConfig
from core.provenance import (
    ProvenanceManifest,
    compute_sha256,
    export_provenance_manifest,
    generate_provenance_manifest,
    get_environment_metadata,
    get_git_commit_hash,
)


def test_compute_sha256(tmp_path: Path) -> None:
    """Verify SHA-256 computation on test file."""
    test_file = tmp_path / "test_data.txt"
    test_file.write_text("IonosphericScintillation Provenance Test", encoding="utf-8")

    sha = compute_sha256(test_file)
    assert isinstance(sha, str)
    assert len(sha) == 64  # SHA-256 hex string length

    with pytest.raises(FileNotFoundError):
        compute_sha256(tmp_path / "non_existent_file.txt")


def test_get_environment_metadata() -> None:
    """Verify environment metadata returns python version and key packages."""
    env = get_environment_metadata()
    assert "python_version" in env
    assert "platform" in env
    assert "numpy" in env
    assert "scipy" in env
    assert env["numpy"] != "not installed"


def test_get_git_commit_hash() -> None:
    """Verify git commit hash returns valid string or fallback."""
    git_hash = get_git_commit_hash()
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0


def test_generate_provenance_manifest(tmp_path: Path) -> None:
    """Verify manifest generation with config, repair logs, and file checksum."""
    sample_file = tmp_path / "sample.pm6"
    sample_file.write_bytes(b"Header data\n100 200 300\n")

    cfg = ProcessingConfig(mtm_nw=3.5, mtm_n_tapers=6)
    repair_logs = ["Repaired 1 non-monotonic timestamp at index 12"]

    manifest = generate_provenance_manifest(
        input_file_path=sample_file,
        config=cfg,
        repair_logs=repair_logs,
    )

    assert isinstance(manifest, ProvenanceManifest)
    assert manifest.app_version == "1.0.0"
    assert manifest.input_file["filename"] == "sample.pm6"
    assert "sha256" in manifest.input_file
    assert manifest.config["mtm_nw"] == 3.5
    assert manifest.repair_logs == repair_logs

    d = manifest.to_dict()
    assert d["manifest_version"] == "1.0"
    assert d["config"]["mtm_n_tapers"] == 6

    json_str = manifest.to_json()
    parsed = json.loads(json_str)
    assert parsed["app_version"] == "1.0.0"


def test_export_provenance_manifest(tmp_path: Path) -> None:
    """Verify manifest export writes formatted JSON file to disk."""
    sample_file = tmp_path / "input.txt"
    sample_file.write_text("Data content", encoding="utf-8")

    output_manifest = tmp_path / "manifests" / "audit_manifest.json"

    result_path = export_provenance_manifest(
        output_path=output_manifest,
        input_file_path=sample_file,
        config=ProcessingConfig(),
    )

    assert result_path.exists()
    content = json.loads(result_path.read_text(encoding="utf-8"))
    assert content["manifest_version"] == "1.0"
    assert content["input_file"]["filename"] == "input.txt"
