import argparse
import json
import pandas as pd
import networkx as nx

from config import DATA_DIR, OUTPUT_DIR


def clean_region_name(region: str) -> str:
    """
    Return a cleaned region name string.
    Keeps names like 'CP_Left' intact, but safely handles missing values.
    """
    if pd.isna(region):
        return ""
    return str(region).strip()


def add_region_node_if_missing(G: nx.DiGraph, region_name: str):
    """
    Add a fallback node if a region is present in the edges file
    but missing from regions.csv.
    """
    if region_name not in G:
        G.add_node(
            region_name,
            region_id="unknown",
            name=region_name,
            x=0.0,
            y=0.0,
            parent="",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build a full region-to-region connectome graph from analyzed edges."
    )

    parser.add_argument(
        "--region",
        choices=["GPe", "GPi", "STN", "SNr"],
        required=True,
        help="Region family / dataset to build from"
    )

    parser.add_argument(
        "--topk_per_source",
        type=int,
        default=0,
        help="Keep only the top-K outgoing edges per source. Use 0 to keep all edges."
    )

    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.0,
        help="Optional minimum edge weight threshold"
    )

    parser.add_argument(
        "--min_fraction",
        type=float,
        default=0.0,
        help="Optional minimum normalized weight threshold"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load atlas regions
    # ----------------------------
    regions_path = DATA_DIR / "regions.csv"
    edges_path = OUTPUT_DIR / f"edges_{args.region}.csv"

    if not regions_path.exists():
        raise FileNotFoundError(f"Missing regions file: {regions_path}")

    if not edges_path.exists():
        raise FileNotFoundError(
            f"Missing edges file: {edges_path}\n"
            f"Run 1_analyze_data.py first."
        )

    regions = pd.read_csv(regions_path)
    edges = pd.read_csv(edges_path)

    required_edge_cols = {
        "source", "target", "weight", "weight_norm", "weight_rank_pct", "metric"
    }
    if not required_edge_cols.issubset(edges.columns):
        raise ValueError(
            f"Edges file missing columns: {required_edge_cols - set(edges.columns)}"
        )

    # ----------------------------
    # Clean and filter edges
    # ----------------------------
    edges = edges.dropna(subset=["source", "target", "weight"]).copy()
    edges["source"] = edges["source"].apply(clean_region_name)
    edges["target"] = edges["target"].apply(clean_region_name)

    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")
    edges["weight_norm"] = pd.to_numeric(edges["weight_norm"], errors="coerce")
    edges["weight_rank_pct"] = pd.to_numeric(edges["weight_rank_pct"], errors="coerce")

    edges = edges.dropna(subset=["weight", "weight_norm", "weight_rank_pct"]).copy()
    edges = edges[edges["weight"] > 0].copy()

    # Optional thresholds
    if args.min_weight > 0:
        edges = edges[edges["weight"] >= args.min_weight].copy()

    if args.min_fraction > 0:
        edges = edges[edges["weight_norm"] >= args.min_fraction].copy()

    # Keep top-K outgoing edges per source if requested
    if args.topk_per_source > 0:
        edges = (
            edges.sort_values(["source", "weight"], ascending=[True, False])
            .groupby("source", as_index=False, group_keys=False)
            .head(args.topk_per_source)
            .copy()
        )

    if edges.empty:
        raise ValueError("No edges remain after filtering.")

    # ----------------------------
    # Build graph
    # ----------------------------
    G = nx.DiGraph()

    # Add atlas nodes from regions.csv
    # We create Left/Right variants too, since your connectome now uses hemi-specific names.
    for _, r in regions.iterrows():
        acr = str(r["acronym"]).strip()

        base_attrs = {
            "region_id": str(r.get("region_id", "unknown")) if pd.notna(r.get("region_id", None)) else "unknown",
            "name": str(r.get("name", acr)) if pd.notna(r.get("name", None)) else acr,
            "x": float(r.get("x_2d", 0.0)) if pd.notna(r.get("x_2d", None)) else 0.0,
            "y": float(r.get("y_2d", 0.0)) if pd.notna(r.get("y_2d", None)) else 0.0,
            "parent": str(r.get("parent_acronym", "")) if pd.notna(r.get("parent_acronym", None)) else "",
        }

        # Add base acronym node
        G.add_node(acr, **base_attrs)

        # Add hemisphere-specific node variants
        G.add_node(f"{acr}_Left", **base_attrs)
        G.add_node(f"{acr}_Right", **base_attrs)

    # Add edges from analyzed connectome
    for _, e in edges.iterrows():
        src = e["source"]
        tgt = e["target"]

        # Some region-hemi names may not exist in regions.csv, so create fallback nodes if needed
        add_region_node_if_missing(G, src)
        add_region_node_if_missing(G, tgt)

        G.add_edge(
            src,
            tgt,
            weight=float(e["weight"]),
            weight_norm=float(e["weight_norm"]),
            rank_pct=float(e["weight_rank_pct"]),
            metric=str(e["metric"]),
        )

    # ----------------------------
    # Save graph files
    # ----------------------------
    graph_json = nx.node_link_data(G)

    json_path = OUTPUT_DIR / f"graph_{args.region}.json"
    graphml_path = OUTPUT_DIR / f"graph_{args.region}.graphml"

    json_path.write_text(json.dumps(graph_json, indent=2))
    #    # GraphML cannot handle None values, so replace them with safe defaults
    #for node, attrs in G.nodes(data=True):
     #   for key, value in list(attrs.items()):
      #      if value is None:
       #         if key in {"x", "y"}:
        #            attrs[key] = 0.0
         #       else:
          #          attrs[key] = ""

    #for u, v, attrs in G.edges(data=True):
     ###         attrs[key] = ""
    #nx.write_graphml(G, graphml_path)

    print(f"\n[Build] Dataset: {args.region}")
    print(f"Saved graph JSON → {json_path}")
    print(f"Saved graph GraphML → {graphml_path}")
    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()