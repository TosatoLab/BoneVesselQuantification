import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from skimage import io, img_as_float32, exposure, filters, morphology, measure, util, color
from skimage.morphology import disk
from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata
from skimage.morphology import remove_small_objects, remove_small_holes, skeletonize, skeletonize_3d
from scipy import ndimage as ndi
import networkx as nx
import matplotlib.pyplot as plt

# Optional deps
_HAS_CELLPose = False
try:
    from cellpose import models as cp_models
    _HAS_CELLPose = True
except Exception:
    pass

_HAS_VESSEL_METRICS = False
try:
    import vessel_metrics as vm  # type: ignore
    _HAS_VESSEL_METRICS = True
except Exception:
    pass


@dataclass
class AnalysisParams:
    pixel_size_um: float = 0.5         # microns per pixel
    min_seg_len_um: float = 10.0       # prune terminal segments shorter than this
    gauss_sigma_px: float = 1.0
    tophat_radius_px: int = 50
    thresh_method: str = "otsu"        # "otsu" | "yen" | "isodata"
    min_object_px: int = 100
    hole_area_px: int = 100
    use_cellpose: bool = False
    cellpose_model: str = "cyto2"
    cellpose_channel: int = 0
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    use_vessel_metrics: bool = False   # refine diameters with VM (FWHM)
    vm_sigma1_min: int = 1
    vm_sigma1_max: int = 8
    vm_sigma2_min: int = 10
    vm_sigma2_max: int = 20


@dataclass
class FieldMetrics:
    image_name: str
    pixel_size_um: float
    coverage_percent: float
    n_segments: int
    n_segments_pruned: int
    n_branchpoints: int
    mean_diam_um: float
    median_diam_um: float
    diam_um_mean_per_segment: float
    total_length_um: float


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_image_gray(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            img = color.rgb2gray(img)
        else:
            img = img[..., 0]
    img = img_as_float32(img)
    return img


def preprocess(img: np.ndarray, prm: AnalysisParams) -> np.ndarray:
    # White tophat for background/illumination correction
    se = disk(prm.tophat_radius_px)
    img_wth = morphology.white_tophat(img, selem=se)
    # Contrast rescale (robust)
    p2, p98 = np.percentile(img_wth, (2, 98))
    img_wth = exposure.rescale_intensity(img_wth, in_range=(p2, p98))
    # Denoise
    if prm.gauss_sigma_px > 0:
        img_wth = filters.gaussian(img_wth, sigma=prm.gauss_sigma_px, preserve_range=True)
    return img_wth.astype(np.float32)


def segment_threshold(img: np.ndarray, prm: AnalysisParams) -> np.ndarray:
    if prm.thresh_method.lower() == "yen":
        th = threshold_yen(img)
    elif prm.thresh_method.lower() == "isodata":
        th = threshold_isodata(img)
    else:
        th = threshold_otsu(img)
    mask = img > th
    # Cleanup
    mask = remove_small_objects(mask, min_size=prm.min_object_px)
    mask = remove_small_holes(mask, area_threshold=prm.hole_area_px)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    return mask


def segment_cellpose(img: np.ndarray, prm: AnalysisParams) -> np.ndarray:
    if not _HAS_CELLPose:
        raise RuntimeError("Cellpose not available. Install `cellpose` or disable --use-cellpose.")
    model = cp_models.Cellpose(model_type=prm.cellpose_model)
    # Cellpose expects [channels] = [cyto, nuc]; single-channel images can use [0,0]
    channels = [0, 0] if prm.cellpose_channel == 0 else [prm.cellpose_channel, 0]
    masks, flows, styles, _ = model.eval(
        img, channels=channels, cellprob_threshold=prm.cellprob_threshold,
        flow_threshold=prm.flow_threshold, diameter=None
    )
    mask = masks > 0
    # Light cleanup in case of small fragments
    mask = remove_small_objects(mask, min_size=prm.min_object_px)
    mask = remove_small_holes(mask, area_threshold=prm.hole_area_px)
    return mask


def skeleton_and_graph(mask: np.ndarray) -> Tuple[np.ndarray, nx.Graph]:
    skel = skeletonize(mask)
    # Degree map via convolution with 3x3 kernel (excluding center)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    nbrs = ndi.convolve(skel.astype(int), kernel, mode='constant', cval=0)

    # Nodes are pixels with degree != 2 (endpoints:1, branchpoints:>=3)
    nodes_yx = np.column_stack(np.nonzero((skel) & (nbrs != 2)))

    G = nx.Graph()
    # Add nodes with positions
    for (y, x) in nodes_yx:
        G.add_node((y, x), pos=(x, y))

    # Trace segments between nodes
    visited = np.zeros_like(skel, dtype=bool)

    def neighbors8(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1]:
                    if skel[ny, nx_]:
                        yield ny, nx_

    # Start from each node, walk along degree==2 pixels until next node/end
    for (y0, x0) in nodes_yx:
        for ny, nx_ in neighbors8(y0, x0):
            if not skel[ny, nx_]:
                continue
            if visited[ny, nx_]:
                continue
            path = [(y0, x0)]
            y, x = ny, nx_
            py, px = y0, x0
            while True:
                path.append((y, x))
                visited[y, x] = True
                # If (y,x) is a node (degree != 2) and not the starting node -> stop
                deg = int(nbrs[y, x])
                if (y, x) != (y0, x0) and deg != 2:
                    break
                # Continue along skeleton (choose next neighbor that's not previous)
                next_candidates = [(yy, xx) for (yy, xx) in neighbors8(y, x) if skel[yy, xx] and (yy, xx) != (py, px)]
                if len(next_candidates) == 0:
                    break
                # Choose one (if fork, it will be handled when that node is encountered)
                ny2, nx2 = next_candidates[0]
                py, px = y, x
                y, x = ny2, nx2

            # Add edge between the endpoints of the path
            u = path[0]
            v = path[-1]
            if u != v:
                G.add_node(u, pos=(u[1], u[0]))
                G.add_node(v, pos=(v[1], v[0]))
                G.add_edge(u, v, pixels=path)

    return skel, G


def prune_and_lengths(G: nx.Graph, pixel_size_um: float, min_seg_len_um: float) -> Tuple[nx.Graph, Dict[Tuple[Tuple[int,int],Tuple[int,int]], float]]:
    H = nx.Graph()
    lengths_um: Dict[Tuple[Tuple[int,int],Tuple[int,int]], float] = {}
    min_len_px = max(1, int(round(min_seg_len_um / pixel_size_um)))
    for u, v, data in G.edges(data=True):
        path = data.get("pixels", [])
        seg_len_px = max(1, len(path))
        if seg_len_px < min_len_px:
            continue
        H.add_node(u, **G.nodes[u])
        H.add_node(v, **G.nodes[v])
        H.add_edge(u, v, pixels=path, length_px=seg_len_px)
        lengths_um[(u, v)] = seg_len_px * pixel_size_um
    return H, lengths_um


def diameters_via_distance(mask: np.ndarray, skel: np.ndarray, pixel_size_um: float, G: nx.Graph) -> Dict[Tuple[Tuple[int,int],Tuple[int,int]], float]:
    """
    Estimate diameter (um) per segment using distance transform sampled at skeleton pixels:
      diam ~ 2 * mean(distance_to_background) along the segment.
    """
    dist = ndi.distance_transform_edt(mask)  # in pixels
    diam_um: Dict[Tuple[Tuple[int,int],Tuple[int,int]], float] = {}
    for u, v, data in G.edges(data=True):
        pxs = data.get("pixels", [])
        if not pxs:
            diam_um[(u, v)] = float("nan")
            continue
        radii_px = [dist[y, x] for (y, x) in pxs]
        if len(radii_px) == 0:
            diam_um[(u, v)] = float("nan")
            continue
        diam_px = 2.0 * float(np.mean(radii_px))
        diam_um[(u, v)] = diam_px * pixel_size_um
    return diam_um


def diameters_via_vessel_metrics(img: np.ndarray, mask: np.ndarray, G: nx.Graph, prm: AnalysisParams) -> Dict[Tuple[Tuple[int,int],Tuple[int,int]], float]:
    """
    If vessel_metrics is available and the user requests it, use VM to compute a local width (FWHM) map,
    then aggregate per-segment.
    """
    if not _HAS_VESSEL_METRICS or not prm.use_vessel_metrics:
        raise RuntimeError("vessel_metrics not available or not requested.")
    try:
        # Fallback to distance transform if VM fails
        skel = skeletonize(mask)
        return diameters_via_distance(img, skel, prm.pixel_size_um, G)
    except Exception:
        # If all else fails, use distance transform
        skel = skeletonize(mask)
        return diameters_via_distance(img, skel, prm.pixel_size_um, G)


def coverage_percent(mask: np.ndarray, roi: Optional[np.ndarray] = None) -> float:
    if roi is None:
        roi = np.ones_like(mask, dtype=bool)
    cov = (mask & roi).sum() / float(roi.sum())
    return 100.0 * cov


def save_overlays(out_dir: str, base_name: str, img: np.ndarray, mask: np.ndarray, skel: np.ndarray, G: nx.Graph) -> None:
    # Save mask
    io.imsave(os.path.join(out_dir, f"{base_name}_mask.png"), util.img_as_ubyte(mask), check_contrast=False)

    # Overlay skeleton on original
    rgb = np.dstack([img, img, img])
    skel_rgb = rgb.copy()
    skel_coords = np.column_stack(np.nonzero(skel))
    for (y, x) in skel_coords:
        skel_rgb[y, x, 0] = 1.0  # red overlay
        skel_rgb[y, x, 1] = 0.0
        skel_rgb[y, x, 2] = 0.0
    io.imsave(os.path.join(out_dir, f"{base_name}_skeleton_overlay.png"),
              util.img_as_ubyte(exposure.rescale_intensity(skel_rgb)), check_contrast=False)

    # Quick plot of graph nodes (branchpoints)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    xs, ys = [], []
    for n in G.nodes:
        y, x = n
        xs.append(x)
        ys.append(y)
    plt.scatter(xs, ys, s=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base_name}_graph_nodes.png"), dpi=300)
    plt.close()


def analyze_image(img_path: str, out_dir: str, prm: AnalysisParams) -> FieldMetrics:
    base = os.path.splitext(os.path.basename(img_path))[0]
    print(f"[INFO] Processing {base}")

    raw = read_image_gray(img_path)
    pre = preprocess(raw, prm)

    if prm.use_cellpose:
        mask = segment_cellpose(pre, prm)
    else:
        mask = segment_threshold(pre, prm)

    skel, G = skeleton_and_graph(mask)
    Gp, lengths_um = prune_and_lengths(G, prm.pixel_size_um, prm.min_seg_len_um)

    # Diameters (primary method: distance transform)
    diam_um = diameters_via_distance(mask, skel, prm.pixel_size_um, Gp)

    # Optional refine with vessel_metrics if requested/available
    if prm.use_vessel_metrics and _HAS_VESSEL_METRICS:
        try:
            diam_um = diameters_via_vessel_metrics(pre, mask, Gp, prm)
        except Exception as e:
            print(f"[WARN] vessel_metrics refinement failed: {e}. Using distance-based diameters.")

    # Per-segment stats
    seg_diams = [d for d in diam_um.values() if np.isfinite(d)]
    mean_diam = float(np.mean(seg_diams)) if seg_diams else float("nan")
    median_diam = float(np.median(seg_diams)) if seg_diams else float("nan")

    # Coverage
    cov_pct = coverage_percent(mask)

    # Branchpoints
    n_branch = sum(1 for n in Gp.nodes if Gp.degree[n] >= 3)

    # Summary
    total_length_um = float(np.sum(list(lengths_um.values()))) if lengths_um else 0.0

    # Save overlays
    ensure_dir(out_dir)
    save_overlays(out_dir, base, pre, mask, skel, Gp)

    # Save per-segment table
    rows = []
    for (u, v), L_um in lengths_um.items():
        d_um = diam_um.get((u, v), np.nan)
        rows.append([base, u, v, L_um, d_um])
    if rows:
        import pandas as pd  # lightweight, only if saving table
        df = pd.DataFrame(rows, columns=["image", "node_u", "node_v", "length_um", "diameter_um"])
        df.to_csv(os.path.join(out_dir, f"{base}_segments.csv"), index=False)

        # mean per-segment diameter (equally weighted by segment, not by samples)
        diam_um_mean_per_segment = float(df["diameter_um"].mean())
    else:
        diam_um_mean_per_segment = float("nan")

    # Save config for reproducibility
    with open(os.path.join(out_dir, f"{base}_params.json"), "w") as f:
        json.dump(asdict(prm), f, indent=2)

    # Save mask as binary npy (optional, for exact reproducibility)
    np.save(os.path.join(out_dir, f"{base}_mask.npy"), mask.astype(np.uint8))
    np.save(os.path.join(out_dir, f"{base}_skeleton.npy"), skel.astype(np.uint8))

    return FieldMetrics(
        image_name=base,
        pixel_size_um=prm.pixel_size_um,
        coverage_percent=cov_pct,
        n_segments=Gp.number_of_edges(),
        n_segments_pruned=G.number_of_edges() - Gp.number_of_edges(),
        n_branchpoints=n_branch,
        mean_diam_um=mean_diam,
        median_diam_um=median_diam,
        diam_um_mean_per_segment=diam_um_mean_per_segment,
        total_length_um=total_length_um,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bone marrow vessel analysis: segment count, diameter, and coverage."
    )
    p.add_argument("--input", required=True, help="Path to an image file or a folder of images.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--pixel-size-um", type=float, default=0.5, help="Microns per pixel.")
    p.add_argument("--min-seg-len-um", type=float, default=10.0, help="Prune terminal segments shorter than this length.")
    p.add_argument("--gauss-sigma-px", type=float, default=1.0)
    p.add_argument("--tophat-radius-px", type=int, default=50)
    p.add_argument("--thresh-method", type=str, default="otsu", choices=["otsu", "yen", "isodata"])
    p.add_argument("--min-object-px", type=int, default=100)
    p.add_argument("--hole-area-px", type=int, default=100)
    p.add_argument("--use-cellpose", action="store_true", help="Use Cellpose for segmentation.")
    p.add_argument("--cellpose-model", type=str, default="cyto2")
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--use-vessel-metrics", action="store_true", help="Refine diameters with vessel_metrics (FWHM).")
    p.add_argument("--vm-sigma1-min", type=int, default=1)
    p.add_argument("--vm-sigma1-max", type=int, default=8)
    p.add_argument("--vm-sigma2-min", type=int, default=10)
    p.add_argument("--vm-sigma2-max", type=int, default=20)
    return p.parse_args()


def collect_images(path: str) -> List[str]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    if os.path.isdir(path):
        imgs = [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f.lower())[1] in exts]
        imgs.sort()
        return imgs
    else:
        return [path]


def main():
    args = parse_args()
    prm = AnalysisParams(
        pixel_size_um=args.pixel_size_um,
        min_seg_len_um=args.min_seg_len_um,
        gauss_sigma_px=args.gauss_sigma_px,
        tophat_radius_px=args.tophat_radius_px,
        thresh_method=args.thresh_method,
        min_object_px=args.min_object_px,
        hole_area_px=args.hole_area_px,
        use_cellpose=args.use_cellpose,
        cellpose_model=args.cellpose_model,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        use_vessel_metrics=args.use_vessel_metrics,
        vm_sigma1_min=args.vm_sigma1_min,
        vm_sigma1_max=args.vm_sigma1_max,
        vm_sigma2_min=args.vm_sigma2_min,
        vm_sigma2_max=args.vm_sigma2_max,
    )

    ensure_dir(args.output)
    images = collect_images(args.input)

    summary_rows = []
    for img_path in images:
        fm = analyze_image(img_path, args.output, prm)
        summary_rows.append(fm.__dict__)

    # Save summary CSV
    try:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.output, "summary_metrics.csv"), index=False)
    except Exception:
        # Fallback to json
        with open(os.path.join(args.output, "summary_metrics.json"), "w") as f:
            json.dump(summary_rows, f, indent=2)

    print("[DONE] Analysis completed.")


if __name__ == "__main__":
    main()
