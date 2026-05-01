import argparse
import pandas as pd

import glob
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR


def build_pathways_from_edges(edges_path: str, source: str) -> pd.DataFrame:
    """Original mode: edges.csv with columns source,target,weight."""
    edges = pd.read_csv(edges_path)

    required = {"source", "target", "weight"}
    if not required.issubset(edges.columns):
        raise ValueError(f"edges file missing columns: {required - set(edges.columns)}")

    edges = edges.dropna(subset=["source", "target", "weight"]).copy()
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    edges = edges.dropna(subset=["weight"])
    edges = edges[edges["weight"] > 0].copy()

    out = edges[edges["source"] == source].copy()
    if out.empty:
        raise ValueError(f"No outgoing edges for '{source}' found in {edges_path}")

    pathways = out.groupby("target", as_index=False)["weight"].sum()
    pathways = pathways[pathways["weight"] > 0].copy()

    total = pathways["weight"].sum()
    if total <= 0:
        raise ValueError(f"Total outgoing weight for '{source}' is 0.")

    pathways["fraction"] = pathways["weight"] / total
    pathways = pathways.sort_values("fraction", ascending=False).reset_index(drop=True)
    pathways["rank"] = pathways.index + 1
    pathways.insert(0, "source", source)
    return pathways[["source", "target", "weight", "fraction", "rank"]]


def build_full_connectome_from_swc_summary(
    swc_path: str,
    source_col: str,
    target_col: str,
    weight_col: str,
    exclude_outside_brain: bool,
    exclude_targets: list[str],
) -> pd.DataFrame:
    """
    Build FULL region-to-region connectome:
    start_region_hemi -> end_region_hemi

    No forced source labeling.
    Preserves true biological connectivity.
    """

    df = pd.read_csv(swc_path)

    # ----------------------------
    # Validate columns
    # ----------------------------
    required = {source_col, target_col, weight_col}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required - set(df.columns)}"
        )

    # ----------------------------
    # Clean data
    # ----------------------------
    df = df.dropna(subset=[source_col, target_col, weight_col]).copy()

    df[source_col] = df[source_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[weight_col])
    df = df[df[weight_col] > 0]

    # ----------------------------
    # Remove unwanted regions
    # ----------------------------
    if exclude_outside_brain:
        df = df[
            ~df[source_col].str.contains("outside_brain", case=False, na=False) &
            ~df[target_col].str.contains("outside_brain", case=False, na=False)
        ]

    if exclude_targets:
        df = df[~df[target_col].isin(exclude_targets)]

    # Always remove root (critical)
    df = df[
        ~df[source_col].str.contains("root", case=False, na=False) &
        ~df[target_col].str.contains("root", case=False, na=False)
    ]

    # ----------------------------
    # Aggregate connections
    # ----------------------------
    pathways = df.groupby(
        [source_col, target_col],
        as_index=False
    )[weight_col].sum()

    pathways = pathways.rename(columns={
        source_col: "source",
        target_col: "target",
        weight_col: "weight"
    })

    # ----------------------------
    # Normalize per source
    # ----------------------------
    pathways["fraction"] = pathways.groupby("source")["weight"] \
        .transform(lambda x: x / x.sum())

    # Rank within each source
    pathways["rank"] = pathways.groupby("source")["fraction"] \
        .rank(ascending=False, method="first").astype(int)

    return pathways.sort_values(["source", "fraction"], ascending=[True, False])
def pathways_to_edges(pathways: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Convert pathways_{source}.csv style output into edges_{source}.csv
    expected by 2_build_circuit.py.
    """
    edges = pathways.copy()

    # Normalize weights for convenience (build uses weight + optional fields)
    edges["weight_norm"] = edges["fraction"]

    # rank_pct: 1.0 for rank 1, down to ~0 for last
    n = len(edges)
    edges["weight_rank_pct"] = 1.0 if n <= 1 else 1.0 - (edges["rank"] - 1) / (n - 1)

    edges["metric"] = metric
    return edges[["source", "target", "weight", "weight_norm", "weight_rank_pct", "metric"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", default=None,
                    help="Full source region name, e.g. GPe_Left")
    parser.add_argument("--region", choices=["GPe", "GPi", "STN", "SNr"], default=None,
                        help="Region family to use for auto file selection")
    parser.add_argument("--side", choices=["Left", "Right"], default=None,
                        help="Hemisphere side for auto source naming")

    # Original mode input (edges)
    parser.add_argument("--edges", default=str(DATA_DIR / "edges.csv"),
                        help="Edges CSV with columns source,target,weight")

    # New SWC summary mode input
    parser.add_argument("--swc_summary", default=None,
                        help="SWC region summary CSV (e.g., data/swc_region_summary.csv). "
                             "If omitted, the newest swc_region_summary*.csv in data/ is used.")

    # How to interpret SWC summary columns
    parser.add_argument("--swc_source_col", default="start_region_hemi",
                        help="Column to use as source (default: start_region_hemi)")
    parser.add_argument("--swc_target_col", default="end_region_hemi",
                        help="Column to use as target (default: end_region_hemi)")
    parser.add_argument("--swc_weight_col", default="length_um",
                        help="Column to use as weight (default: length_um)")

    # Cleanup flags for SWC data
    parser.add_argument("--exclude_outside_brain", action="store_true",
                        help="Exclude rows where source/target == 'outside_brain' (recommended)")
    parser.add_argument("--exclude_targets", default="",
                        help="Comma-separated list of targets to drop (exact match), e.g. 'alv_Right,ccb_Right'")

    args = parser.parse_args()
    REGION_TO_FILE = {
    "GPe": DATA_DIR / "gpeswc_region_summary.csv",
    "GPi": DATA_DIR / "gpi_swc_region_summary.csv",
    "STN": DATA_DIR / "stn_swc_region_summary.csv",
    "SNr": DATA_DIR / "snr_swc_region_summary.csv",
}

    if args.region and args.side:
        source = f"{args.region}_{args.side}"
    elif args.region:
        source = args.region #whole brain datasets often have hemisphere-specific region names (e.g. GPe_Left, GPe_Right) but the user might specify source as "GPe". In that case, we can auto-clean the region names to ignore hemisphere for matching.
    elif args.source:
        source = args.source
    else:
        raise ValueError("Provide either --source or both --region and --side.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Decide SWC file to use
    if args.swc_summary:
        swc_path = args.swc_summary
    elif args.region:
        swc_path = REGION_TO_FILE[args.region]
        print(f"[Region-Selected SWC Summary] Using: {swc_path}")
    else:
        pattern = str(DATA_DIR / "swc_region_summary*.csv")
        files = glob.glob(pattern)
        if not files:
            raise ValueError("No swc_region_summary*.csv files found in data directory.")
        swc_path = max(files, key=lambda f: Path(f).stat().st_mtime)
        print(f"[Auto-Detected SWC Summary] Using: {swc_path}")

    # Build pathways (SWC mode)
    exclude_targets = [x.strip() for x in args.exclude_targets.split(",") if x.strip()]
    pathways = build_full_connectome_from_swc_summary(
        swc_path=swc_path,
        source_col=args.swc_source_col,
        target_col=args.swc_target_col,
        weight_col=args.swc_weight_col,
        exclude_outside_brain=args.exclude_outside_brain,
        exclude_targets=exclude_targets,
    )

    out_path = OUTPUT_DIR / f"pathways_{source}.csv"
    pathways.to_csv(out_path, index=False)
    # Save edges_<source>.csv so build step can read it
    edges_df = pathways_to_edges(pathways, metric=args.swc_weight_col)
    edges_path = OUTPUT_DIR / f"edges_{source}.csv"
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved edges -> {edges_path}")

    print(f"\n[Analyze] Source: {source}")
    print(f"Mode: SWC summary ({swc_path})")
    print(f"Saved pathways -> {out_path}")
    print("\nTop pathways:")
    print(pathways.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
