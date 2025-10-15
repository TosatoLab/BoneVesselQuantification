# BoneVesselQuantification

Title: Bone Marrow Vessel Analysis — segments, diameters, coverage

CLI tool to quantify bone marrow vascular architecture from confocal images (MIP or single plane). 
Uses scikit-image + networkx by default; optionally cellpose (segmentation) and vessel_metrics (FWHM diameters).

Installation
# create & activate a clean environment (recommended)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip

# core deps
pip install numpy scipy scikit-image networkx matplotlib pandas

# optional (for tougher segmentations)
pip install cellpose

# optional (for refined diameters)
pip install vessel-metrics  # package name may vary; see the library docs


Usage:
python bm_vessel_analysis.py \
  --input path/to/images_or_image.tif \
  --output results/ \
  --pixel-size-um 0.5 \
  --min-seg-len-um 10 \
  --thresh-method otsu

# with Cellpose segmentation (recommended when signal is variable)
python bm_vessel_analysis.py \
  --input path/to/folder \
  --output results_cellpose \
  --use-cellpose --cellpose-model cyto2 \
  --pixel-size-um 0.5

# attempt vessel_metrics-based diameter refinement (falls back automatically if not available)
python bm_vessel_analysis.py \
  --input path/to/folder \
  --output results_vm \
  --use-vessel-metrics

  
Outputs per image:

<name>_mask.png — binary vessel mask

<name>_skeleton_overlay.png — QA overlay (skeleton in red on preprocessed image)

<name>_graph_nodes.png — node visualization

<name>_segments.csv — per-segment length (µm) and diameter (µm)

<name>_params.json — frozen parameters for reproducibility

(repo-level) summary_metrics.csv — one row per image with: coverage (%), segment counts, branchpoints, mean/median diameter, total network length

Notes & caveats

Pixel size matters: set --pixel-size-um correctly (µm/px) from your microscope metadata.

Pruning: terminal segments shorter than --min-seg-len-um are removed (noise/gaps).

Diameter method: by default, diameter is 2 × distance_to_background sampled at skeleton. This is standard and reproducible; if you need FWHM along cross-lines, enable --use-vessel-metrics and install the package.

2D vs 3D: script expects 2D images or MIPs; full 3D requires skeletonize_3d + 3D graphing.
