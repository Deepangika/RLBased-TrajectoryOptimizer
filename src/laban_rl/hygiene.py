"""Credential and generated-artifact hygiene checks without secret disclosure."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any


CREDENTIAL_ENV_NAMES = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
SENSITIVE_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*api_key*",
    "credentials.json",
    "secrets.json",
)
_CREDENTIAL_PATTERNS = (
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("generic_api_key", re.compile(r"\b(?:sk|key)-[0-9A-Za-z_-]{16,}\b")),
)


def _tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [
        root / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def credential_presence_check(root: str | Path) -> dict[str, Any]:
    """Report credential locations/types only, never credential values."""
    repository = Path(root).resolve()
    tracked_findings: list[dict[str, str]] = []
    for path in _tracked_files(repository):
        try:
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = str(path.relative_to(repository))
        for finding_type, pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(text):
                tracked_findings.append(
                    {"path": relative, "type": finding_type}
                )
    local_sensitive = []
    for pattern in SENSITIVE_FILE_PATTERNS:
        for path in repository.glob(pattern):
            if path.is_file() and path.name not in (".env.example",):
                local_sensitive.append(str(path.relative_to(repository)))
    return {
        "configured_environment_variables": [
            name for name in CREDENTIAL_ENV_NAMES if os.environ.get(name)
        ],
        "tracked_credential_findings": tracked_findings,
        "local_sensitive_files": sorted(set(local_sensitive)),
        "safe": not tracked_findings and not local_sensitive,
    }
