import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CARE-Depth Stage 6
# CARE-Depth V2 Reliability-Gated Temporal Experiment
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FOCAL_LENGTH = 520.0
BASELINE = 0.12

NUM_DISPARITIES = 128
BLOCK_SIZE = 5

SCENARIOS = [
    "normal",
    "low_texture",
    "illumination_change",
    "sensor_noise",
    "moving_pedestrian",
    "combined",
]

NUM_FRAMES = 5

print("=" * 70)
print("CARE-Depth Stage 6 - CARE-Depth V2 Experiment")
print("=" * 70)


# ------------------------------------------------------------
# Stereo matcher
# ------------------------------------------------------------

def create_matcher():

    return cv2.StereoSGBM_create(
        minDisparity=0,
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
# Create stereo pair
# ------------------------------------------------------------

def create_stereo_pair(image, ground_truth, frame_number):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Small frame-to-frame illumination variation
    illumination_factor = 1.0 + 0.025 * np.sin(frame_number)

    gray = np.clip(
        gray.astype(np.float32) * illumination_factor,
        0,
        255
    ).astype(np.uint8)

    # Small sensor variation
    rng = np.random.default_rng(1000 + frame_number)

    noise = rng.normal(
        0,
        3.0,
        gray.shape
    )

    left = np.clip(
        gray.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    # Physical stereo relationship
    safe_depth = np.maximum(
        ground_truth,
        0.5
    )

    disparity = (
        FOCAL_LENGTH
        * BASELINE
        / safe_depth
    )

    h, w = gray.shape

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

    return left, right


# ------------------------------------------------------------
# Stereo depth
# ------------------------------------------------------------

def stereo_depth(left, right):

    matcher = create_matcher()

    disparity = (
        matcher.compute(left, right)
        .astype(np.float32)
        / 16.0
    )

    valid = disparity > 0.5

    depth = np.zeros_like(disparity)

    depth[valid] = (
        FOCAL_LENGTH
        * BASELINE
        / disparity[valid]
    )

    return disparity, depth, valid


# ------------------------------------------------------------
# Texture reliability
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

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    local_texture = cv2.GaussianBlur(
        magnitude,
        (15, 15),
        0
    )

    scale = np.percentile(
        local_texture,
        95
    ) + 1e-6

    return np.clip(
        local_texture / scale,
        0,
        1
    )


# ------------------------------------------------------------
# Edge reliability
# ------------------------------------------------------------

def edge_score(image):

    edges = cv2.Canny(
        image,
        50,
        150
    )

    edges = edges.astype(
        np.float32
    ) / 255.0

    density = cv2.GaussianBlur(
        edges,
        (11, 11),
        0
    )

    return np.clip(
        density / (np.max(density) + 1e-6),
        0,
        1
    )


# ------------------------------------------------------------
# Temporal reliability
# ------------------------------------------------------------

def temporal_score(current, previous):

    if previous is None:
        return np.ones_like(
            current,
            dtype=np.float32
        )

    score = np.ones_like(
        current,
        dtype=np.float32
    )

    valid = (
        (current > 0)
        & (previous > 0)
    )

    difference = np.abs(
        current - previous
    )

    score[valid] = np.exp(
        -difference[valid] / 0.50
    )

    return np.clip(
        score,
        0,
        1
    )


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate(estimate, ground_truth):

    valid = (
        (estimate > 0)
        & np.isfinite(estimate)
        & np.isfinite(ground_truth)
    )

    coverage = (
        np.sum(valid)
        / ground_truth.size
        * 100
    )

    if np.sum(valid) == 0:
        return {
            "coverage": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "relative_error": np.nan,
            "accuracy_10": 0,
            "accuracy_25": 0,
        }

    error = np.abs(
        estimate[valid]
        - ground_truth[valid]
    )

    return {
        "coverage": coverage,
        "mae": np.mean(error),
        "rmse": np.sqrt(np.mean(error ** 2)),
        "relative_error": np.mean(
            error / ground_truth[valid]
        ),
        "accuracy_10": np.mean(
            error <= 0.10
        ) * 100,
        "accuracy_25": np.mean(
            error <= 0.25
        ) * 100,
    }


# ------------------------------------------------------------
# CARE temporal fusion
# ------------------------------------------------------------

def care_fusion(
    current,
    previous,
    previous_previous,
    reliability
):

    # Start with the current stereo estimate.
    care = current.copy()

    # No temporal information is available yet.
    if previous is None:
        return care

    # Build the temporal estimate.
    if previous_previous is None:

        temporal_estimate = (
            0.60 * current
            + 0.40 * previous
        )

    else:

        temporal_estimate = (
            0.50 * current
            + 0.30 * previous
            + 0.20 * previous_previous
        )

    # --------------------------------------------------------
    # CARE-Depth V2: reliability-gated temporal fusion
    #
    # High reliability:
    #     trust the current stereo estimate.
    #
    # Lower reliability:
    #     allow a limited amount of temporal information
    #     to contribute.
    #
    # The temporal contribution is capped at 30 percent.
    # --------------------------------------------------------

    temporal_weight = np.zeros_like(
        reliability,
        dtype=np.float32
    )

    # Very reliable current estimates:
    # keep the current stereo estimate unchanged.
    high_reliability = reliability >= 0.50
    temporal_weight[high_reliability] = 0.0

    # Moderately reliable estimates:
    # increase temporal contribution gradually as
    # reliability decreases from 0.50 to 0.25.
    moderate_reliability = (
        (reliability >= 0.25)
        & (reliability < 0.50)
    )

    temporal_weight[moderate_reliability] = (
        0.30
        * (
            0.50
            - reliability[moderate_reliability]
        )
        / 0.25
    )

    # Low-reliability estimates:
    # allow the maximum bounded temporal contribution.
    low_reliability = reliability < 0.25
    temporal_weight[low_reliability] = 0.30

    # Keep the fusion bounded and numerically safe.
    temporal_weight = np.clip(
        temporal_weight,
        0.0,
        0.30
    )

    # Reliability-gated fusion.
    care = (
        (1.0 - temporal_weight) * current
        + temporal_weight * temporal_estimate
    )

    # Invalid current pixels use temporal information
    # only as a fallback.
    invalid = current <= 0

    care[invalid] = temporal_estimate[invalid]

    return care


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

ground_truth = np.load(
    os.path.join(
        DATA_DIR,
        "stage3_ground_truth_depth.npy"
    )
)

results = []

best_visualization = None

for scenario in SCENARIOS:

    print("\n" + "-" * 70)
    print("Scenario:", scenario)

    image_path = os.path.join(
        DATA_DIR,
        f"scenario_{scenario}.png"
    )

    image = cv2.imread(image_path)

    if image is None:
        print("Image missing:", image_path)
        continue

    previous_depth = None
    previous_previous_depth = None

    frame_baseline = []
    frame_care = []

    for frame in range(NUM_FRAMES):

        left, right = create_stereo_pair(
            image,
            ground_truth,
            frame
        )

        disparity, current_depth, valid = (
            stereo_depth(
                left,
                right
            )
        )

        texture = texture_score(left)
        edge = edge_score(left)
        temporal = temporal_score(
            current_depth,
            previous_depth
        )

        # ----------------------------------------------------
        # Reliability model
        #
        # R = 0.35C + 0.25T + 0.20E + 0.20S
        #
        # C is approximated from disparity validity.
        # ----------------------------------------------------

        consistency = valid.astype(
            np.float32
        )

        reliability = (
            0.35 * consistency
            + 0.25 * texture
            + 0.20 * edge
            + 0.20 * temporal
        )

        reliability[~valid] = 0

        care_depth = care_fusion(
            current_depth,
            previous_depth,
            previous_previous_depth,
            reliability
        )

        baseline_metrics = evaluate(
            current_depth,
            ground_truth
        )

        care_metrics = evaluate(
            care_depth,
            ground_truth
        )

        frame_baseline.append(
            baseline_metrics
        )

        frame_care.append(
            care_metrics
        )

        previous_previous_depth = (
            None
            if previous_depth is None
            else previous_depth.copy()
        )

        previous_depth = current_depth.copy()

        if scenario == "combined" and frame == NUM_FRAMES - 1:

            best_visualization = {
                "image": image,
                "stereo": current_depth,
                "care": care_depth,
                "reliability": reliability,
            }

    # Average over the five temporal frames
    baseline_mae = np.mean([
        x["mae"] for x in frame_baseline
    ])

    care_mae = np.mean([
        x["mae"] for x in frame_care
    ])

    baseline_rmse = np.mean([
        x["rmse"] for x in frame_baseline
    ])

    care_rmse = np.mean([
        x["rmse"] for x in frame_care
    ])

    baseline_rel = np.mean([
        x["relative_error"]
        for x in frame_baseline
    ])

    care_rel = np.mean([
        x["relative_error"]
        for x in frame_care
    ])

    baseline_acc10 = np.mean([
        x["accuracy_10"]
        for x in frame_baseline
    ])

    care_acc10 = np.mean([
        x["accuracy_10"]
        for x in frame_care
    ])

    baseline_acc25 = np.mean([
        x["accuracy_25"]
        for x in frame_baseline
    ])

    care_acc25 = np.mean([
        x["accuracy_25"]
        for x in frame_care
    ])

    baseline_coverage = np.mean([
        x["coverage"]
        for x in frame_baseline
    ])

    care_coverage = np.mean([
        x["coverage"]
        for x in frame_care
    ])

    print(
        f"Baseline MAE : {baseline_mae:.4f} m"
    )

    print(
        f"CARE MAE     : {care_mae:.4f} m"
    )

    print(
        f"Baseline RMSE: {baseline_rmse:.4f} m"
    )

    print(
        f"CARE RMSE    : {care_rmse:.4f} m"
    )

    print(
        f"MAE change   : "
        f"{baseline_mae - care_mae:+.4f} m"
    )

    results.append({

        "scenario": scenario,

        "baseline_MAE_m":
            baseline_mae,

        "CARE_MAE_m":
            care_mae,

        "baseline_RMSE_m":
            baseline_rmse,

        "CARE_RMSE_m":
            care_rmse,

        "baseline_relative_error":
            baseline_rel,

        "CARE_relative_error":
            care_rel,

        "baseline_accuracy_10cm_%":
            baseline_acc10,

        "CARE_accuracy_10cm_%":
            care_acc10,

        "baseline_accuracy_25cm_%":
            baseline_acc25,

        "CARE_accuracy_25cm_%":
            care_acc25,

        "baseline_coverage_%":
            baseline_coverage,

        "CARE_coverage_%":
            care_coverage,
    })


# ------------------------------------------------------------
# Save table
# ------------------------------------------------------------

df = pd.DataFrame(results)

csv_path = os.path.join(
    RESULTS_DIR,
    "stage6_care_depth_v2_results.csv"
)

df.to_csv(
    csv_path,
    index=False
)

txt_path = os.path.join(
    RESULTS_DIR,
    "stage6_care_depth_v2_results.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARE-Depth Stage 6 CARE-Depth V2 Temporal Experiment\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Five-frame temporal evaluation was performed "
        "independently for each hospital scenario.\n\n"
    )

    f.write(
        "Reliability equation:\n"
        "R = 0.35C + 0.25T + 0.20E + 0.20S\n\n"
    )

    f.write(
        "CARE-Depth V2 temporal fusion:\n"
        "Z_CARE = R*Z_current + (1-R)*Z_temporal\n\n"
    )

    f.write(
        df.to_string(index=False)
    )


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

if best_visualization is not None:

    stereo = best_visualization["stereo"]
    care = best_visualization["care"]
    reliability = best_visualization["reliability"]

    plt.figure(figsize=(9, 6))
    plt.imshow(stereo)
    plt.colorbar(label="Depth (m)")
    plt.title(
        "Conventional Stereo Depth - Combined Scenario"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "stage6_combined_stereo.png"
        ),
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.imshow(care)
    plt.colorbar(label="Depth (m)")
    plt.title(
        "CARE-Depth Estimate - Combined Scenario"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "stage6_combined_care.png"
        ),
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.imshow(
        reliability,
        vmin=0,
        vmax=1
    )
    plt.colorbar(label="Reliability")
    plt.title(
        "CARE-Depth Reliability Map - Combined Scenario"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "stage6_combined_reliability.png"
        ),
        dpi=200
    )
    plt.close()


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal temporal results:")
print(df.to_string(index=False))

print("\nSaved:")
print(csv_path)
print(txt_path)

print("\nFigures saved to:")
print(FIGURES_DIR)