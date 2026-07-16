"""Build the prospectively frozen independent R21 4/4/8 founder panel."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from .build_founder_bank import R4_PRODUCTION_SPEC, run_founder_bank_builder


PANEL_SEED = 9007
SCREENING_SEEDS = (12_000, 12_001)
SCREENING_HORIZON = 7_500
SELECTION_RULE = "ascending-first-qualified-clone-r21-independent-v1"


def main() -> None:
    artifact_root = Path(__file__).with_name("r21_artifacts")
    spec = replace(
        R4_PRODUCTION_SPEC,
        seeds=SCREENING_SEEDS,
        horizon=SCREENING_HORIZON,
        selection_rule_id=SELECTION_RULE,
    )
    result = run_founder_bank_builder(
        smoke=False,
        bank_directory=artifact_root / "founders",
        screening_record_path=artifact_root / "founder_screening.jsonl",
        configured_spec=spec,
        panel_seed=PANEL_SEED,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
