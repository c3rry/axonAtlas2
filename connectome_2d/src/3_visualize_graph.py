import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath

from config import OUTPUT_DIR, FIGURES_DIR, DATA_DIR


def clean_region_name(name):
    return (
        str(name)
        .replace("_Left", "")
        .replace("_Right", "")
        .replace(" (L)", "")
        .replace(" (R)", "")
    )
# ----------------------------
# Geometry helpers
# ----------------------------
def pol2xy(r: float, theta_rad: float):
    return (r * math.cos(theta_rad), r * math.sin(theta_rad))


def deg2rad(d: float):
    return math.radians(d)


def arc_points(theta_start_deg: float, theta_end_deg: float, r: float, n: int = 24):
    """
    Sample points along an arc from theta_start_deg to theta_end_deg.
    """
    pts = []
    for i in range(n + 1):
        t = theta_start_deg + (theta_end_deg - theta_start_deg) * (i / n)
        pts.append(pol2xy(r, deg2rad(t)))
    return pts


# ----------------------------
# Label formatting
# ----------------------------
def format_region_label(region: str) -> str:
    """
    Convert:
      PR_Left  -> PR (L)
      VL_Right -> VL (R)
    """
    if not isinstance(region, str):
        return str(region)

    hemi = None
    if region.endswith("_Left"):
        hemi = "L"
        core = region[:-5]
    elif region.endswith("_Right"):
        hemi = "R"
        core = region[:-6]
    else:
        core = region

    core = core.replace("_", " ").strip()
    return f"{core} ({hemi})" if hemi else core


# ----------------------------
# Region colors
# ----------------------------
def load_region_colors(labels_path: Path) -> dict[str, str]:
    """
    Load Allen-style region colors from labels.csv and map:
      acronym
      acronym_Left
      acronym_Right
    all to the same atlas color.
    """
    if not labels_path.exists():
        return {}

    df = pd.read_csv(labels_path)
    region_col, color_col = "acronym", "color_hex_triplet"

    if region_col not in df.columns or color_col not in df.columns:
        raise ValueError(f"labels file must contain '{region_col}' and '{color_col}'")

    color_map = {}
    for _, row in df.iterrows():
        acr = str(row[region_col]).strip()
        hex_triplet = str(row[color_col]).strip()

        if not acr or acr.lower() == "nan":
            continue
        if not hex_triplet or hex_triplet.lower() == "nan":
            continue

        color = "#" + hex_triplet.lstrip("#").lower()
        color_map[acr] = color
        color_map[f"{acr}_Left"] = color
        color_map[f"{acr}_Right"] = color

    return color_map


# ----------------------------
# Data preparation
# ----------------------------
def clean_connectome_df(
    pathways_df: pd.DataFrame,
    min_weight: float = 0.0,
    min_fraction: float = 0.0,
    topk_per_source: int = 0,
):
    """
    Clean and filter the full connectome table.

    Expected columns:
      source, target, weight, fraction, rank
    """
    required = {"source", "target", "weight", "fraction", "rank"}
    if not required.issubset(pathways_df.columns):
        raise ValueError(f"Missing columns: {required - set(pathways_df.columns)}")

    df = pathways_df.copy()

    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")

    df = df.dropna(subset=["source", "target", "weight", "fraction"]).copy()
    df = df[df["weight"] > 0].copy()

    # Remove non-biological buckets entirely
    exclude_patterns = ("root", "outside_brain")
    for col in ["source", "target"]:
        df = df[
            ~df[col].str.lower().str.contains("|".join(exclude_patterns), na=False)
        ].copy()

    if min_weight > 0:
        df = df[df["weight"] >= min_weight].copy()

    if min_fraction > 0:
        df = df[df["fraction"] >= min_fraction].copy()

    if df.empty:
        raise ValueError("No rows remain after filtering.")

    # Keep top-K outgoing targets per source if requested
    if topk_per_source > 0:
        df = (
            df.sort_values(["source", "weight"], ascending=[True, False])
            .groupby("source", as_index=False, group_keys=False)
            .head(topk_per_source)
            .copy()
        )

    # Recompute fractions and ranks per source after filtering
    df["fraction"] = df.groupby("source")["weight"].transform(lambda x: x / x.sum())
    df["rank"] = (
        df.groupby("source")["fraction"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    return df.sort_values(["source", "weight"], ascending=[True, False]).reset_index(drop=True)


def trim_to_top_regions(df: pd.DataFrame, top_regions: int = None) -> pd.DataFrame:
    """
    Keep only the top regions by total incident connectivity
    (outgoing + incoming), then keep only edges whose source and target
    are both in that set.

    This helps keep the circular plot readable.
    """
    if top_regions <= 0:
        return df.copy()

    source_totals = df.groupby("source")["weight"].sum()
    target_totals = df.groupby("target")["weight"].sum()

    all_regions = sorted(set(source_totals.index) | set(target_totals.index))
    totals = {}
    for region in all_regions:
        totals[region] = float(source_totals.get(region, 0.0)) + float(target_totals.get(region, 0.0))

    keep = pd.Series(totals).sort_values(ascending=False).head(top_regions).index
    keep = set(keep)

    out = df[df["source"].isin(keep) & df["target"].isin(keep)].copy()
    if out.empty:
        raise ValueError("No edges remain after top-region trimming.")

    # Recompute per-source fractions after trimming
    out["fraction"] = out.groupby("source")["weight"].transform(lambda x: x / x.sum())
    out["rank"] = (
        out.groupby("source")["fraction"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return out.reset_index(drop=True)


def compute_region_sizes(df: pd.DataFrame) -> pd.Series:
    """
    Region arc size is based on total incident weight:
      outgoing + incoming
    """
    source_totals = df.groupby("source")["weight"].sum()
    target_totals = df.groupby("target")["weight"].sum()

    regions = sorted(set(source_totals.index) | set(target_totals.index))
    sizes = {}
    for region in regions:
        sizes[region] = float(source_totals.get(region, 0.0)) + float(target_totals.get(region, 0.0))

    s = pd.Series(sizes).sort_values(ascending=False)
    if (s <= 0).all():
        raise ValueError("All region sizes are zero.")
    return s


def ordered_regions_for_circle(region_sizes: pd.Series) -> list[str]:
    """
    Arrange larger regions in an alternating sequence to spread them around
    the circle and reduce crowding a bit.
    """
    ordered = list(region_sizes.index)
    left = ordered[::2]
    right = ordered[1::2][::-1]
    return left + right


def compute_arc_angles(region_sizes: pd.Series, gap_deg: float = 2.2, start_deg: float = 90.0):
    """
    Allocate circle arc spans proportional to region_sizes.
    """
    regions = ordered_regions_for_circle(region_sizes)
    total_size = float(region_sizes.loc[regions].sum())

    n = len(regions)
    full = 360.0
    total_gap = gap_deg * n
    available = full - total_gap
    if available <= 0:
        raise ValueError("Too many regions / too much gap for arc allocation.")

    arc_angles = {}
    angle = start_deg

    for region in regions:
        frac = float(region_sizes[region]) / total_size
        span = available * frac
        arc_angles[region] = (angle, angle - span)  # clockwise
        angle -= span + gap_deg

    return arc_angles, regions


def allocate_region_edge_segments(df: pd.DataFrame, arc_angles: dict[str, tuple[float, float]]):
    """
    Allocate a slice on each region arc for every edge, once for outgoing usage
    and once for incoming usage.

    This lets ribbons connect exact source slice -> target slice.
    """
    segments = {}

    # Outgoing allocation along source arcs
    for source, sub in df.groupby("source", sort=False):
        a1, a2 = arc_angles[source]
        span = abs(a1 - a2)
        total_out = sub["weight"].sum()

        cursor = a1
        for idx, row in sub.sort_values("weight", ascending=False).iterrows():
            frac = row["weight"] / total_out
            seg_span = span * frac
            seg_a1 = cursor
            seg_a2 = cursor - seg_span
            segments.setdefault(idx, {})
            segments[idx]["source_seg"] = (seg_a1, seg_a2)
            cursor = seg_a2

    # Incoming allocation along target arcs
    for target, sub in df.groupby("target", sort=False):
        a1, a2 = arc_angles[target]
        span = abs(a1 - a2)
        total_in = sub["weight"].sum()

        cursor = a1
        for idx, row in sub.sort_values("weight", ascending=False).iterrows():
            frac = row["weight"] / total_in
            seg_span = span * frac
            seg_a1 = cursor
            seg_a2 = cursor - seg_span
            segments.setdefault(idx, {})
            segments[idx]["target_seg"] = (seg_a1, seg_a2)
            cursor = seg_a2

    return segments


# ----------------------------
# Ribbon drawing
# ----------------------------
def chord_ribbon_patch(
    src_a1: float,
    src_a2: float,
    tgt_a1: float,
    tgt_a2: float,
    r_anchor: float = 0.93,
    r_ctrl: float = 0.20,
    facecolor: str = "#999999",
    alpha: float = 0.62,
    edgecolor: str = "black",
    lw: float = 0.35,
):
    """
    Build a smooth chord ribbon from a source arc slice to a target arc slice.
    Ribbon is colored by the source region.
    """
    src_arc = arc_points(src_a1, src_a2, r_anchor, n=14)
    tgt_arc = arc_points(tgt_a1, tgt_a2, r_anchor, n=14)

    S1 = src_arc[0]
    S2 = src_arc[-1]
    T1 = tgt_arc[0]
    T2 = tgt_arc[-1]

    C1 = pol2xy(r_ctrl, deg2rad(src_a1))
    C2 = pol2xy(r_ctrl, deg2rad(tgt_a1))
    C3 = pol2xy(r_ctrl, deg2rad(tgt_a2))
    C4 = pol2xy(r_ctrl, deg2rad(src_a2))

    verts = []
    codes = []

    # Start at source outer slice start
    verts.append(S1)
    codes.append(MplPath.MOVETO)

    # Curve source -> target
    verts.extend([C1, C2, T1])
    codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

    # Walk target arc
    for p in tgt_arc[1:]:
        verts.append(p)
        codes.append(MplPath.LINETO)

    # Curve target -> source
    verts.extend([C3, C4, S2])
    codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

    # Walk source arc backwards
    for p in reversed(src_arc[:-1]):
        verts.append(p)
        codes.append(MplPath.LINETO)

    verts.append(S1)
    codes.append(MplPath.CLOSEPOLY)

    return PathPatch(
        MplPath(verts, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        alpha=alpha,
        zorder=1,
    )


# ----------------------------
# Labels
# ----------------------------
def spread_labels(items, min_gap=0.07, y_min=-1.28, y_max=1.28):
    """
    Spread label y-positions to reduce overlaps.
    items = [(region, y), ...]
    """
    if not items:
        return []

    ys = [max(y_min, min(y, y_max)) for _, y in items]

    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap

    if ys[-1] > y_max:
        ys[-1] = y_max
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)

    return [(items[i][0], ys[i]) for i in range(len(items))]


def add_label_with_leader(ax, region: str, arc_angles: dict, radius: float = 1.0, fontsize: int = 9):
    """
    Add leader-line labels outside the circle.
    """
    a1, a2 = arc_angles[region]
    mid = (a1 + a2) / 2

    x0, y0 = pol2xy(radius + 0.02, deg2rad(mid))
    x1, y1 = pol2xy(radius + 0.10, deg2rad(mid))

    if x0 >= 0:
        x_text = 1.33
        ha = "left"
    else:
        x_text = -1.33
        ha = "right"

    ax.plot([x0, x1, x_text], [y0, y1, y1], color="black", linewidth=0.65, zorder=4)
    ax.text(
        x_text,
        y1,
        format_region_label(region),
        ha=ha,
        va="center",
        fontsize=fontsize,
        zorder=5,
        clip_on=False,
    )


# ----------------------------
# Main circular plot
# ----------------------------
def region_to_group(region: str) -> str:
    base = str(region).replace("_Left", "").replace("_Right", "").strip()

    # Remove junk
    if base.lower() in ("root", "outside_brain"):
        return None

    # Exact mappings
    exact = {
        "CP": "Striatum",
        "ACB": "Striatum",
        "FS": "Striatum",
        "STR": "Striatum",

        "GPe": "Pallidum",
        "GPi": "Pallidum",
        "SI": "Pallidum",
        "PAL": "Pallidum",

        "SNr": "Midbrain",
        "VTA": "Midbrain",
        "PAG": "Midbrain",
        "SCig": "Midbrain",
        "APN": "Midbrain",
        "MB": "Midbrain",

        "PO": "Thalamus",
        "MD": "Thalamus",
        "VL": "Thalamus",
        "VPM": "Thalamus",
        "VM": "Thalamus",
        "CL": "Thalamus",
        "LP": "Thalamus",
        "AD": "Thalamus",
        "PR": "Thalamus",
        "ZI": "Thalamus",

        "BLAa": "Amygdala",

        "EPv": "Pallidum",

        "SF": "Septal Nuclei",

        "alv": "Fiber Tracts",
        "fi": "Fiber Tracts",
        "int": "Fiber Tracts",
        "ml": "Fiber Tracts",
        "scp": "Fiber Tracts",
        "nst": "Fiber Tracts",
        "opt": "Fiber Tracts",


        "DMH": "Hypothalamus",
        "VMH": "Hypothalamus",
        "LPO": "Hypothalamus",
        "MPO": "Hypothalamus",
        "HY": "Hypothalamus",

        "P": "Hindbrain",
        "MY": "Hindbrain",
    }

    if base in exact:
        return exact[base]

    # Cortex families
    cortex_prefix = (
        "MOp", "MOs", "SSp", "SSs", "VIS", "AUD",
        "RSP", "ORB", "ACA", "AI", "ILA", "PL",
        "FRP", "PTLp", "TEa", "ECT", "PERI",
        "VISC", "GU"
    )
    if base.startswith(cortex_prefix):
        return "Cerebral Cortex"

    # Hippocampus
    if base.startswith(("CA1", "CA2", "CA3", "DG", "SUB", "ENT")):
        return "Hippocampal Formation"

    # Olfactory
    if base.startswith(("AON", "PIR", "OLF")):
        return "Olfactory Areas"

    return "Unmapped"

def compute_group_distribution(df: pd.DataFrame):
    group_weights = {}
    unmapped = set()

    for _, row in df.iterrows():
        group = region_to_group(row["target"])

        if group is None:
            continue

        if group == "Unmapped":
            base = row["target"].replace("_Left", "").replace("_Right", "")
            unmapped.add(base)

        group_weights[group] = group_weights.get(group, 0) + row["weight"]

    total = sum(group_weights.values())

    group_percent = {
        k: (v / total) * 100 for k, v in group_weights.items()
    }

    return group_percent, unmapped

def draw_full_circular_connectome(
    pathways_df: pd.DataFrame,
    out_path: str,
    region_name: str,
    min_weight: float = 0.0,
    min_fraction: float = 0.0,
    topk_per_source: int = 0,
    top_regions: int = 18,
    show_table: bool = True,
    table_rows: int = 16,
):
    """
    Draw a full circular region-to-region chord diagram.
    """
    df = clean_connectome_df(
        pathways_df,
        min_weight=min_weight,
        min_fraction=min_fraction,
        topk_per_source=topk_per_source,
    )

    df = trim_to_top_regions(df, top_regions=top_regions)

    region_sizes = compute_region_sizes(df)
    arc_angles, ordered_regions = compute_arc_angles(region_sizes, gap_deg=2.2, start_deg=92.0)
    edge_segments = allocate_region_edge_segments(df, arc_angles)

    labels_path = DATA_DIR / "labels.csv"
    region_colors = load_region_colors(labels_path)
    fallback_color = "#cccccc"

    # Figure
    if show_table:
        fig = plt.figure(figsize=(12, 14), facecolor="white")
        gs = fig.add_gridspec(2, 1, height_ratios=[8.7, 4.3], hspace=0.04)
        ax = fig.add_subplot(gs[0, 0])
        ax_tbl = fig.add_subplot(gs[1, 0])
        ax_tbl.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
        ax_tbl = None

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)

    radius = 1.0
    arc_width = 0.08
    r_anchor = 0.92
    r_ctrl = 0.18

    # Draw ribbons first
    for idx, row in df.sort_values("weight", ascending=False).iterrows():
        src = row["source"]
        tgt = row["target"]

        src_seg = edge_segments[idx]["source_seg"]
        tgt_seg = edge_segments[idx]["target_seg"]

        ribbon = chord_ribbon_patch(
            src_seg[0],
            src_seg[1],
            tgt_seg[0],
            tgt_seg[1],
            r_anchor=r_anchor,
            r_ctrl=r_ctrl,
            facecolor=region_colors.get(src, fallback_color),  # ribbon matches source color
            alpha=0.58,
            edgecolor="black",
            lw=0.28,
        )
        ax.add_patch(ribbon)

    # Draw outer arcs
    for region in ordered_regions:
        a1, a2 = arc_angles[region]
        ax.add_patch(
            Wedge(
                (0, 0),
                radius,
                theta1=a2,
                theta2=a1,
                width=arc_width,
                facecolor=region_colors.get(region, fallback_color),
                edgecolor="black",
                linewidth=0.3,
                zorder=3,
            )
        )
    for region in ordered_regions:
        a1, a2 = arc_angles[region]
        ax.add_patch(
            Wedge(
                (0, 0),
                radius,
                theta1=a2,
                theta2=a1,
                width=arc_width,
                facecolor=region_colors.get(region, fallback_color),
                edgecolor="black",
                linewidth=1.0,
                zorder=3,
            )
        )
    # ----------------------------
    # Highlight TARGET regions (inner band)
    # ----------------------------
    target_regions = set(df["target"])

    for region in ordered_regions:
        if region not in target_regions:
            continue

        a1, a2 = arc_angles[region]

        ax.add_patch(
            Wedge(
                (0, 0),
                radius - arc_width * 0.15,   # slightly inside main arc
                theta1=a2,
                theta2=a1,
                width=arc_width * 0.25,      # thin band
                facecolor="white",           # highlight color
                edgecolor="none",
                alpha=0.9,
                zorder=4,
            )
        )

    # Labels
    right_items = []
    left_items = []

    for region in ordered_regions:
        a1, a2 = arc_angles[region]
        mid = (a1 + a2) / 2
        x, y = pol2xy(radius + 0.10, deg2rad(mid))
        if x >= 0:
            right_items.append((region, y))
        else:
            left_items.append((region, y))

    right_items.sort(key=lambda z: z[1])
    left_items.sort(key=lambda z: z[1])

    right_spread = spread_labels(right_items)
    left_spread = spread_labels(left_items)

    def add_spread_label(region, y_override):
        a1, a2 = arc_angles[region]
        mid = (a1 + a2) / 2

        x0, y0 = pol2xy(radius + 0.02, deg2rad(mid))
        x1, y1 = pol2xy(radius + 0.10, deg2rad(mid))

        if x0 >= 0:
            x_text = 1.32
            ha = "left"
        else:
            x_text = -1.32
            ha = "right"

        ax.plot([x0, x1, x_text], [y0, y1, y_override], color="black", linewidth=0.65, zorder=4)
        ax.text(
            x_text,
            y_override,
            format_region_label(region),
            ha=ha,
            va="center",
            fontsize=9,
            zorder=5,
            clip_on=False,
        )

    for region, y in right_spread:
        add_spread_label(region, y)

    for region, y in left_spread:
        add_spread_label(region, y)

    ax.text(
        0,
        1.42,
        f"Region-to-Region Connectome ({region_name})",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # ----------------------------
    # Pie Chart (Grouped Output)
    # ----------------------------
    if show_table and ax_tbl is not None:
        group_percent, unmapped = compute_group_distribution(df)

        # Sort slices from largest to smallest, but keep Unmapped last if present
        items = sorted(group_percent.items(), key=lambda x: x[1], reverse=True)

        if any(k == "Unmapped" for k, _ in items):
            unmapped_item = [x for x in items if x[0] == "Unmapped"]
            items = [x for x in items if x[0] != "Unmapped"] + unmapped_item

        labels = [k for k, _ in items]
        sizes = [v for _, v in items]

        # Colors (match your example style)
        def group_to_color(group):
            sample_regions = {
                "Striatum": "CP",
                "Cerebral Cortex": "MOp",
                "Midbrain": "MB",
                "Thalamus": "TH",
                "Hindbrain": "MY",
                "Hypothalamus": "HY",
                "Pallidum": "GPe",
                "Hippocampal Formation": "CA1",
                "Olfactory Areas": "PIR",
            }
            
            base = sample_regions.get(group, None)
            return region_colors.get(base, "#CCCCCC")

        colors = [group_to_color(g) for g in labels]


        wedges, texts = ax_tbl.pie(
        sizes,
        labels=None,
        startangle=90,
        colors=colors,
        radius=1.2
        )
        # Add percentage labels:
        # - large slices get labels inside
        # - small slices get labels outside with leader lines
        small_threshold = 4.0  # percent

        inside_labels = []
        outside_right = []
        outside_left = []

        for wedge, pct in zip(wedges, sizes):
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            angle_rad = math.radians(angle)

            x = math.cos(angle_rad)
            y = math.sin(angle_rad)

            label = f"{pct:.1f}%"

            if pct >= small_threshold:
                inside_labels.append((x, y, label))
            else:
                item = {
                    "x": x,
                    "y": y,
                    "label": label,
                    "xy": (1.05 * x, 1.05 * y),
                }
                if x >= 0:
                    outside_right.append(item)
                else:
                    outside_left.append(item)

        # Draw inside labels
        for x, y, label in inside_labels:
            ax_tbl.text(
                0.72 * x,
                0.72 * y,
                label,
                ha="center",
                va="center",
                fontsize=9
            )

        def spread_side(items, min_gap=0.10, y_min=-1.25, y_max=1.25):
            if not items:
                return []

            items = sorted(items, key=lambda d: d["y"])
            ys = [max(y_min, min(d["y"] * 1.45, y_max)) for d in items]

            # forward pass
            for i in range(1, len(ys)):
                if ys[i] - ys[i - 1] < min_gap:
                    ys[i] = ys[i - 1] + min_gap

            # backward pass
            if ys[-1] > y_max:
                ys[-1] = y_max
                for i in range(len(ys) - 2, -1, -1):
                    ys[i] = min(ys[i], ys[i + 1] - min_gap)

            for item, y_new in zip(items, ys):
                item["y_text"] = y_new

            return items

        outside_right = spread_side(outside_right)
        outside_left = spread_side(outside_left)

        # Draw outside labels with leader lines
        for item in outside_right:
            ax_tbl.annotate(
                item["label"],
                xy=item["xy"],
                xytext=(1.55, item["y_text"]),
                ha="left",
                va="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black")
            )

        for item in outside_left:
            ax_tbl.annotate(
                item["label"],
                xy=item["xy"],
                xytext=(-1.55, item["y_text"]),
                ha="right",
                va="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="black")
            )

        ax_tbl.set_title(f"{region_name} Output by Brain Area (%)", fontsize=12)

        # Legend
        legend_labels = [f"{l} ({s:.1f}%)" for l, s in zip(labels, sizes)]
        ax_tbl.legend(
            wedges,
            legend_labels,
            title="Brain Area",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=9
        )

        # ----------------------------
        # Region Mapping Key (NEW)
        # ----------------------------
        group_to_regions = {}

        for _, row in df.iterrows():
            group = region_to_group(row["target"])
            if group is None:
                continue

            base = row["target"].replace("_Left", "").replace("_Right", "")
            group_to_regions.setdefault(group, set()).add(base)

        # Build display text
        mapping_lines = []
        for group in labels:  # keep same order as pie
            regions = sorted(group_to_regions.get(group, []))

            if len(regions) > 6:
                text = ", ".join(regions[:6]) + "..."
            else:
                text = ", ".join(regions)

            mapping_lines.append(f"{group}: {text}")

        # ----------------------------
        # Colored Region Mapping Key
        # ----------------------------
        x_group = -0.95
        x_regions = -0.62
        y_start = 0.22
        line_spacing = 0.085

        for i, group in enumerate(labels):
            regions = sorted(group_to_regions.get(group, []))

            if len(regions) > 6:
                region_text = ", ".join(regions[:6]) + "..."
            else:
                region_text = ", ".join(regions)

            y = y_start - i * line_spacing
            color = group_to_color(group)

            # Group name (colored)
            ax_tbl.text(
                x_group,
                y,
                f"{group}:",
                transform=ax_tbl.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                color=color,
                fontweight="bold"
            )

            # Region list (black)
            ax_tbl.text(
                x_regions,
                y,
                region_text,
                transform=ax_tbl.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                color="black"
            )

        # 🔴 Print unmapped regions
        if unmapped:
            print("\n[WARNING] Unmapped regions detected:")
            for r in sorted(unmapped):
                print(" -", r)

        

    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Draw a circular multi-region connectome chord diagram."
    )
    parser.add_argument(
        "--region",
        choices=["GPe", "GPi", "STN", "SNr"],
        required=True,
        help="Dataset family to visualize",
    )
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--min_fraction", type=float, default=0.0)
    parser.add_argument("--topk_per_source", type=int, default=0)
    parser.add_argument("--top_regions", type=int, default=0)
    parser.add_argument("--table_rows", type=int, default=16)
    parser.add_argument("--no_table", action="store_true")
    args = parser.parse_args()

    pathways_path = OUTPUT_DIR / f"pathways_{args.region}.csv"
    if not pathways_path.exists():
        raise FileNotFoundError(f"Missing {pathways_path}. Run 1_analyze_data.py first.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pathways_path)
    out_path = FIGURES_DIR / f"connectome_circular_{args.region}.png"

    

    draw_full_circular_connectome(
        df,
        str(out_path),
        min_weight=args.min_weight,
        region_name=args.region,
        min_fraction=args.min_fraction,
        topk_per_source=args.topk_per_source,
        top_regions=args.top_regions,
        show_table=not args.no_table,
        table_rows=args.table_rows,
    )

    print(f"[Circular Connectome] Saved figure → {out_path}")


if __name__ == "__main__":
    main()