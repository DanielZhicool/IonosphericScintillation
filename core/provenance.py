"""
Scientific Provenance and Audit Manifest Exporter for Ionospheric Scintillation Analysis.

Generates reproducible audit manifests in JSON format containing SHA-256 data checksums,
git commit hashes, environment library versions, ProcessingConfig parameters, and execution timestamps.
"""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import ProcessingConfig


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hexadecimal hash of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 calculation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash() -> str:
    """Retrieve current git commit SHA-1 hash, or fallback if unavailable."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_environment_metadata() -> dict[str, str]:
    """Retrieve versions of key dependencies and runtime environment."""
    packages = ["numpy", "scipy", "pandas", "pyside6", "pyqtgraph", "ssqueezepy"]
    env: dict[str, str] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in packages:
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = "not installed"
    return env


@dataclass
class ProvenanceManifest:
    """Structured scientific provenance manifest."""

    manifest_version: str = "1.0"
    app_version: str = "1.0.0"
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str = field(default_factory=get_git_commit_hash)
    input_file: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=get_environment_metadata)
    config: dict[str, Any] = field(default_factory=dict)
    repair_logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert manifest to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def generate_provenance_manifest(
    input_file_path: str | Path | None = None,
    config: ProcessingConfig | None = None,
    repair_logs: list[str] | None = None,
) -> ProvenanceManifest:
    """
    Generate a ProvenanceManifest object.

    Args:
        input_file_path: Optional path to the input raw data file.
        config: ProcessingConfig instance used for analysis.
        repair_logs: List of parser warning or data repair log strings.

    Returns:
        Configured ProvenanceManifest instance.
    """
    file_meta: dict[str, Any] = {}
    if input_file_path is not None:
        p = Path(input_file_path)
        if p.is_file():
            file_meta = {
                "path": str(p.resolve()),
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
            }
        elif p.is_dir():
            file_meta = {
                "path": str(p.resolve()),
                "filename": p.name,
                "is_directory": True,
            }
        else:
            file_meta = {"path": str(input_file_path), "status": "file_not_found"}

    cfg_dict = asdict(config) if config is not None else asdict(ProcessingConfig())

    return ProvenanceManifest(
        input_file=file_meta,
        config=cfg_dict,
        repair_logs=repair_logs or [],
    )


def export_provenance_manifest(
    output_path: str | Path,
    input_file_path: str | Path | None = None,
    config: ProcessingConfig | None = None,
    repair_logs: list[str] | None = None,
) -> Path:
    """
    Generate and save provenance manifest to disk as a JSON file.

    Args:
        output_path: File path to save manifest JSON.
        input_file_path: Optional input data file path.
        config: ProcessingConfig instance.
        repair_logs: Optional repair logs.

    Returns:
        Path to written JSON manifest file.
    """
    manifest = generate_provenance_manifest(
        input_file_path=input_file_path,
        config=config,
        repair_logs=repair_logs,
    )
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
    return out_p
