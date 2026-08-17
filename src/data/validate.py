from __future__ import annotations
import pandas as pd
from src.data.clean import RAW_SUM_TOLERANCE, clean_collections

REQUIRED_FIELDS = ["collection_id", "figure_id", "probability", "box_price"]
VALID_RARITIES = {"regular", "rare", "secret"}

NORMALISED_SUM_TOLERANCE = 1e-9

ERROR = "error"
WARNING = "warning"


def _issue(level, check, message, collection_id=None, figure_id=None) -> dict:
    return {
        "level": level,
        "check": check,
        "collection_id": collection_id,
        "figure_id": figure_id,
        "message": message,
    }


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    issues: list[dict] = []
    invalid = pd.Series(False, index=df.index)

    for field in REQUIRED_FIELDS:
        missing = df[field].isna()
        invalid |= missing
        for _, row in df[missing].iterrows():
            issues.append(
                _issue(
                    ERROR,
                    "required_field",
                    f"Missing required field '{field}'",
                    row["collection_id"],
                    row["figure_id"],
                )
            )

    bad_probability = ~df["probability"].between(0, 1) | df["probability"].isna()
    invalid |= bad_probability
    for _, row in df[bad_probability].iterrows():
        issues.append(
            _issue(
                ERROR,
                "probability_range",
                f"probability {row['probability']} outside [0, 1]",
                row["collection_id"],
                row["figure_id"],
            )
        )

    zero_probability = df["probability"] == 0
    for _, row in df[zero_probability].iterrows():
        issues.append(
            _issue(
                WARNING,
                "zero_probability",
                "probability is 0, so this figure can never be pulled",
                row["collection_id"],
                row["figure_id"],
            )
        )

    bad_price = ~(df["box_price"] > 0) | df["box_price"].isna()
    invalid |= bad_price
    for _, row in df[bad_price].iterrows():
        issues.append(
            _issue(
                ERROR,
                "box_price_positive",
                f"box_price {row['box_price']} is not > 0",
                row["collection_id"],
                row["figure_id"],
            )
        )

    duplicated = df.duplicated(subset=["collection_id", "figure_id"], keep=False)
    invalid |= duplicated
    for _, row in df[duplicated].iterrows():
        issues.append(
            _issue(
                ERROR,
                "unique_figure_id",
                "figure_id is not unique within its collection",
                row["collection_id"],
                row["figure_id"],
            )
        )

    unknown_rarity = ~df["rarity"].isin(VALID_RARITIES)
    for _, row in df[unknown_rarity].iterrows():
        issues.append(
            _issue(
                WARNING,
                "rarity_label",
                f"Unrecognised rarity '{row['rarity']}'",
                row["collection_id"],
                row["figure_id"],
            )
        )

    for collection_id, group in df.groupby("collection_id"):
        normalised_sum = group["probability"].sum()
        if abs(normalised_sum - 1) > NORMALISED_SUM_TOLERANCE:
            invalid |= df["collection_id"] == collection_id
            issues.append(
                _issue(
                    ERROR,
                    "probability_sum",
                    f"Normalised probabilities sum to {normalised_sum:.12f}, not 1",
                    collection_id,
                )
            )

        raw_sum = group["probability_raw"].sum()
        if abs(raw_sum - 1) > RAW_SUM_TOLERANCE:
            issues.append(
                _issue(
                    ERROR,
                    "raw_probability_sum",
                    f"Published probabilities sum to {raw_sum:.6f}, "
                    f"more than {RAW_SUM_TOLERANCE} away from 1",
                    collection_id,
                )
            )
            invalid |= df["collection_id"] == collection_id
        elif abs(raw_sum - 1) > 1e-9:
            issues.append(
                _issue(
                    WARNING,
                    "raw_probability_sum",
                    f"Published probabilities sum to {raw_sum:.6f}; "
                    f"rescaled by {1 / raw_sum:.6f} to sum to 1",
                    collection_id,
                )
            )

        if group["box_price"].nunique() > 1:
            issues.append(
                _issue(
                    WARNING,
                    "box_price_consistency",
                    f"Collection has {group['box_price'].nunique()} distinct box prices",
                    collection_id,
                )
            )

        if len(group) < 2:
            issues.append(
                _issue(
                    WARNING,
                    "collection_size",
                    f"Collection has only {len(group)} figure(s)",
                    collection_id,
                )
            )

    df["is_valid"] = ~invalid
    issues_df = pd.DataFrame(
        issues, columns=["level", "check", "collection_id", "figure_id", "message"]
    )
    return df, issues_df


def summarise(issues: pd.DataFrame) -> str:
    if issues.empty:
        return "All validation checks passed."

    lines = []
    for _, issue in issues.iterrows():
        target = " ".join(
            str(part)
            for part in (issue["collection_id"], issue["figure_id"])
            if pd.notna(part) and part
        )
        lines.append(
            f"[{issue['level'].upper()}] {issue['check']} ({target}): {issue['message']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    figures, issues = validate(clean_collections())
    print(summarise(issues))
    print(f"\n{int(figures['is_valid'].sum())}/{len(figures)} figures valid")
