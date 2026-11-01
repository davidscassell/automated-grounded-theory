#!/usr/bin/env python3
"""Write a versioned exp-axial-w1 candidate manifest pinned to this checkout.

The fingerprint scheme matches a startup fingerprint: JSON with
``ensure_ascii=False``, ``sort_keys=True``, ``separators=(",", ":")``,
``default=str``, then sha256. The commit is ``git rev-parse HEAD`` of this
package repository, not the process working directory.

Do not commit a study export, and do not commit Muse ids. ``--source`` adds
an export manifest from a local file. Leave that output uncommitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from automated_grounded_theory.exp_axial_w1 import (
    EXPERIMENT_ID,
    FROZEN_TOKEN_PRICES,
    LEXICAL_ELIGIBILITY_RULE,
    _all_prompts,
    _git_commit,
    _sha,
    startup_fingerprint,
)
from automated_grounded_theory.records import POLICY_VERSION
from automated_grounded_theory.relationship_tiers import PATTERN_MIN_CASES

CANDIDATE_MANIFEST_VERSION = 1


def build_manifest(source_path: Path | None, *, commit: str | None) -> dict:
    pinned = _git_commit() if commit is None else str(commit)
    manifest = {
        "candidate_manifest_version": CANDIDATE_MANIFEST_VERSION,
        "experiment": EXPERIMENT_ID,
        "commit": pinned,
        "fingerprint_scheme": {
            "encoding": "utf-8",
            "json": {
                "ensure_ascii": False,
                "sort_keys": True,
                "separators": [",", ":"],
                "default": "str",
            },
            "digest": "sha256",
        },
        "policy_version": POLICY_VERSION,
        "pattern_min_cases": PATTERN_MIN_CASES,
        "lexical_rule": LEXICAL_ELIGIBILITY_RULE,
        "prompts": _sha(_all_prompts()),
        "eligibility_rules": _sha({
            "pattern_min_cases": PATTERN_MIN_CASES,
            "one_document_one_respondent": False,
            "policy_version": POLICY_VERSION,
            "experiment": EXPERIMENT_ID,
            "lexical_rule": LEXICAL_ELIGIBILITY_RULE,
        }),
        "prices": {
            "input_usd_per_million": FROZEN_TOKEN_PRICES["input_usd_per_million"],
            "output_usd_per_million": FROZEN_TOKEN_PRICES["output_usd_per_million"],
        },
        "export_manifest": None,
        "model_config": None,
        "note": (
            "Pinned to the package repository HEAD. Do not commit a study export or Muse ids. "
            "Pass --source only for a local uncommitted manifest."
        ),
    }
    if source_path is not None:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("--source must be a JSON object.")
        current = startup_fingerprint(payload, "unspecified", commit=pinned)
        manifest["export_manifest"] = current["export_manifest"]
        manifest["model_config"] = current["model_config"]
        manifest["source_name"] = source_path.name
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the manifest here. The default is stdout.")
    parser.add_argument("--source", type=Path, help="Optional local export JSON. Do not commit it or the manifest.")
    parser.add_argument("--commit", help="Override the pinned commit. The default is git rev-parse HEAD.")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.source, commit=args.commit)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
