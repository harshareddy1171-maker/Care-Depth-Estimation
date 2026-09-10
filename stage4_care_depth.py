import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CARE-Depth - Stage 4
# Context-Aware Reliability-Enhanced Depth Estimation
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Camera parameters from Stage 2
FOCAL_LENGTH = 520.0
BASELINE = 0.12

# Stereo parameters
MIN_DISPARITY = 0
NUM_DISPARITIES = 128
BLOCK_SIZE = 5

# Reliability weights
W_CONSISTENCY = 0.35
W_TEXTURE = 0.25
W_EDGE = 0.20
W_TEMPORAL = 0.20

SCENARIOS = [
    "normal",
    "low_texture",
    "illumination_change",
    "sensor_noise",
    "moving_pedestrian",
    "combined",
]

print("=" * 70)
print("CARE-Depth Stage 4 - Reliability + Temporal Fusion")
print("=" * 70)


# ------------------------------------------------------------
# Stereo matcher
# ------------------------------------------------------------

def create_matcher():

    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ------------------------------------------------------------
# Create synthetic stereo pair from a scenario
# ------------------------------------------------------------

def create_stereo_pair(image, ground_truth):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add controlled texture so the stereo algorithm has
    # measurable image information.
    texture = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    texture = texture.astype(np.float32)

    noise = np.random.normal(
        0,
        4.0,
        texture.shape
    )

    left = np.clip(
        texture + noise,
        0,
        255
    ).astype(np.uint8)

    # Theoretical disparity
    safe_depth = np.maximum(ground_truth, 0.5)

    disparity = (
        FOCAL_LENGTH * BASELINE / safe_depth
    )

    h, w = gray.shape

    # Build right image using inverse horizontal mapping.
    x_grid, y_grid = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )

    map_x = (
        x_grid.astype(np.float32)
        + disparity.astype(np.float32)
    )

    map_y = y_grid.astype(np.float32)

    right = cv2.remap(
        left,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return left, right, disparity


# ------------------------------------------------------------
# Conventional stereo depth
# ------------------------------------------------------------

def estimate_stereo_depth(left, right):

    matcher_left = create_matcher()

    disparity_left = (
        matcher_left.compute(left, right)
        .astype(np.float32)
        / 16.0
    )

    # Reverse matcher for left-right consistency
    matcher_right = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity_right = (
        matcher_right.compute(right, left)
        .astype(np.float32)
        / 16.0
    )

    valid = disparity_left > 0.5

    depth = np.zeros_like(disparity_left)

    depth[valid] = (
        FOCAL_LENGTH * BASELINE
        / disparity_left[valid]
    )

    return disparity_left, disparity_right, depth, valid


# ------------------------------------------------------------
# Reliability component 1:
# Left-right consistency
# ------------------------------------------------------------

def consistency_score(disparity_left, disparity_right):

    valid = disparity_left > 0.5

    score = np.zeros_like(disparity_left, dtype=np.float32)

    # A disparity is considered consistent when the
    # forward and reverse estimates agree.
    difference = np.abs(
        disparity_left - disparity_right
    )

    score[valid] = np.exp(
        -difference[valid] / 2.0
    )

    return np.clip(score, 0, 1)


# ------------------------------------------------------------
# Reliability component 2:
# Texture confidence
# ------------------------------------------------------------

def texture_score(image):

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(gx, gy)

    # Local texture estimate
    texture = cv2.GaussianBlur(
        magnitude,
        (15, 15),
        0
    )

    texture = texture / (
        np.percentile(texture, 95) + 1e-6
    )

    return np.clip(texture, 0, 1)


# ------------------------------------------------------------
# Reliability component 3:
# Edge confidence
# ------------------------------------------------------------

def edge_score(image):

    edges = cv2.Canny(
        image,
        50,
        150
    )

    edges = edges.astype(np.float32) / 255.0

    # Spread edge information locally.
    edge_density = cv2.GaussianBlur(
        edges,
        (11, 11),
        0
    )

    edge_density = edge_density / (
        np.max(edge_density) + 1e-6
    )

    return np.clip(edge_density, 0, 1)


# ------------------------------------------------------------
# Reliability component 4:
# Temporal stability
# ------------------------------------------------------------

def temporal_score(current_depth, previous_depth):

    score = np.ones_like(
        current_depth,
        dtype=np.float32
    )

    if previous_depth is None:
        return score

    valid = (
        (current_depth > 0)
        & (previous_depth > 0)
    )

    difference = np.abs(
        current_depth - previous_depth
    )

    # Larger sudden changes receive lower confidence.
    score[valid] = np.exp(
        -difference[valid] / 0.50
    )

    return np.clip(score, 0, 1)


# ------------------------------------------------------------
# CARE-Depth fusion
# ------------------------------------------------------------

def care_fusion(
    current_depth,
    previous_depth,
    previous_previous_depth,
    reliability,
):

    temporal = current_depth.copy()

    if previous_depth is not None:

        temporal = (
            0.50 * current_depth
            + 0.30 * previous_depth
        )

        if previous_previous_depth is not None:

            temporal = (
                0.50 * current_depth
                + 0.30 * previous_depth
                + 0.20 * previous_previous_depth
            )

    care_depth = (
        reliability * current_depth
        + (1.0 - reliability) * temporal
    )

    # If current stereo depth is invalid, use history.
    invalid = current_depth <= 0

    if previous_depth is not None:
        care_depth[invalid] = previous_depth[invalid]

    return care_depth, temporal


# ------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------

def evaluate(estimate, ground_truth):

    valid = (
        np.isfinite(estimate)
        & np.isfinite(ground_truth)
        & (estimate > 0)
        & (ground_truth > 0)
    )

    if np.sum(valid) == 0:

        return {
            "coverage": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "relative_error": np.nan,
            "accuracy_10cm": 0,
            "accuracy_25cm": 0,
        }

    error = np.abs(
        estimate[valid]
        - ground_truth[valid]
    )

    mae = np.mean(error)

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    relative_error = np.mean(
        error / ground_truth[valid]
    )

    accuracy_10cm = np.mean(
        error <= 0.10
    ) * 100

    accuracy_25cm = np.mean(
        error <= 0.25
    ) * 100

    coverage = (
        np.sum(valid)
        / ground_truth.size
        * 100
    )

    return {
        "coverage": coverage,
        "mae": mae,
        "rmse": rmse,
        "relative_error": relative_error,
        "accuracy_10cm": accuracy_10cm,
        "accuracy_25cm": accuracy_25cm,
    }


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

ground_truth = np.load(
    os.path.join(
        DATA_DIR,
        "stage3_ground_truth_depth.npy"
    )
)

all_results = []

previous_depth = None
previous_previous_depth = None

saved_maps = {}

for scenario in SCENARIOS:

    print("\n" + "-" * 70)
    print(f"Processing scenario: {scenario}")

    image_path = os.path.join(
        DATA_DIR,
        f"scenario_{scenario}.png"
    )

    image = cv2.imread(image_path)

    if image is None:
        print("WARNING: scenario image not found.")
        continue

    # Generate stereo pair
    left, right, theoretical_disparity = create_stereo_pair(
        image,
        ground_truth
    )

    # Conventional stereo
    disparity_left, disparity_right, stereo_depth, valid = (
        estimate_stereo_depth(
            left,
            right
        )
    )

    # Reliability components
    consistency = consistency_score(
        disparity_left,
        disparity_right
    )

    texture = texture_score(left)

    edge = edge_score(left)

    temporal = temporal_score(
        stereo_depth,
        previous_depth
    )

    # CARE reliability score
    reliability = (
        W_CONSISTENCY * consistency
        + W_TEXTURE * texture
        + W_EDGE * edge
        + W_TEMPORAL * temporal
    )

    # Do not trust invalid stereo pixels.
    reliability[~valid] = 0.0

    # CARE fusion
    care_depth, temporal_depth = care_fusion(
        stereo_depth,
        previous_depth,
        previous_previous_depth,
        reliability
    )

    # Evaluate both methods
    baseline_metrics = evaluate(
        stereo_depth,
        ground_truth
    )

    care_metrics = evaluate(
        care_depth,
        ground_truth
    )

    print(
        f"Baseline MAE       : "
        f"{baseline_metrics['mae']:.4f} m"
    )

    print(
        f"CARE-Depth MAE     : "
        f"{care_metrics['mae']:.4f} m"
    )

    print(
        f"Baseline RMSE      : "
        f"{baseline_metrics['rmse']:.4f} m"
    )

    print(
        f"CARE-Depth RMSE    : "
        f"{care_metrics['rmse']:.4f} m"
    )

    print(
        f"Mean reliability   : "
        f"{np.mean(reliability[valid]):.4f}"
    )

    # Store results
    all_results.append({
        "scenario": scenario,

        "baseline_coverage_%":
            baseline_metrics["coverage"],

        "baseline_MAE_m":
            baseline_metrics["mae"],

        "baseline_RMSE_m":
            baseline_metrics["rmse"],

        "baseline_relative_error":
            baseline_metrics["relative_error"],

        "baseline_accuracy_10cm_%":
            baseline_metrics["accuracy_10cm"],

        "baseline_accuracy_25cm_%":
            baseline_metrics["accuracy_25cm"],

        "CARE_coverage_%":
            care_metrics["coverage"],

        "CARE_MAE_m":
            care_metrics["mae"],

        "CARE_RMSE_m":
            care_metrics["rmse"],

        "CARE_relative_error":
            care_metrics["relative_error"],

        "CARE_accuracy_10cm_%":
            care_metrics["accuracy_10cm"],

        "CARE_accuracy_25cm_%":
            care_metrics["accuracy_25cm"],

        "mean_reliability":
            np.mean(reliability[valid]),
    })

    # Keep important maps for figures
    if scenario in ["normal", "combined"]:

        saved_maps[scenario] = {
            "image": image,
            "stereo_depth": stereo_depth,
            "care_depth": care_depth,
            "reliability": reliability,
        }

    # Update temporal history
    previous_previous_depth = previous_depth
    previous_depth = stereo_depth.copy()


# ------------------------------------------------------------
# Save numerical results
# ------------------------------------------------------------

results_df = pd.DataFrame(all_results)

csv_path = os.path.join(
    RESULTS_DIR,
    "stage4_care_depth_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

txt_path = os.path.join(
    RESULTS_DIR,
    "stage4_care_depth_results.txt"
)

with open(txt_path, "w", encoding="utf-8") as f:

    f.write("CARE-Depth Stage 4 Results\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        "Reliability model:\n"
        "R = 0.35C + 0.25T + 0.20E + 0.20S\n\n"
    )

    f.write(
        "CARE fusion:\n"
        "Z_CARE = R * Z_current + "
        "(1-R) * Z_temporal\n\n"
    )

    f.write(results_df.to_string(index=False))


# ------------------------------------------------------------
# Generate figures
# ------------------------------------------------------------

for scenario, maps in saved_maps.items():

    image = maps["image"]
    stereo_depth = maps["stereo_depth"]
    care_depth = maps["care_depth"]
    reliability = maps["reliability"]

    # Depth comparison
    plt.figure(figsize=(10, 6))

    plt.imshow(stereo_depth)
    plt.colorbar(label="Depth (m)")
    plt.title(
        f"Conventional Stereo Depth - {scenario}"
    )
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"stage4_{scenario}_stereo_depth.png"
        ),
        dpi=200
    )

    plt.close()

    # CARE depth
    plt.figure(figsize=(10, 6))

    plt.imshow(care_depth)
    plt.colorbar(label="Depth (m)")
    plt.title(
        f"CARE-Depth Estimate - {scenario}"
    )
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"stage4_{scenario}_care_depth.png"
        ),
        dpi=200
    )

    plt.close()

    # Reliability map
    plt.figure(figsize=(10, 6))

    plt.imshow(
        reliability,
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        label="Reliability"
    )

    plt.title(
        f"CARE-Depth Reliability Map - {scenario}"
    )

    plt.xlabel("Pixel")
    plt.ylabel("Pixel")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"stage4_{scenario}_reliability.png"
        ),
        dpi=200
    )

    plt.close()


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults:")
print(results_df.to_string(index=False))

print("\nSaved:")
print(csv_path)
print(txt_path)

print("\nFigures saved in:")
print(FIGURES_DIR)

print("\nCARE-Depth reliability and temporal fusion experiment complete.")