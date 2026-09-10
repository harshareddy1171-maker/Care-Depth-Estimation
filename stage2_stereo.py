import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CARE-Depth - Stage 2
# Physically Consistent Synthetic Stereo Experiment
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. CAMERA PARAMETERS
# ============================================================

WIDTH = 640
HEIGHT = 480

# Synthetic stereo camera parameters
FOCAL_LENGTH = 520.0       # pixels
BASELINE = 0.12            # meters

print("=" * 65)
print("CARE-Depth Stage 2 - Stereo Depth Experiment")
print("=" * 65)

print("\nCamera configuration:")
print(f"Image size       : {WIDTH} x {HEIGHT}")
print(f"Focal length     : {FOCAL_LENGTH:.1f} pixels")
print(f"Stereo baseline  : {BASELINE:.3f} m")


# ============================================================
# 2. LOAD STAGE 1 DATA
# ============================================================

scene_path = DATA_DIR / "hospital_corridor.png"
depth_path = DATA_DIR / "ground_truth_depth.npy"

if not scene_path.exists():
    raise FileNotFoundError(
        f"Missing hospital scene: {scene_path}"
    )

if not depth_path.exists():
    raise FileNotFoundError(
        f"Missing ground-truth depth: {depth_path}"
    )

scene = cv2.imread(str(scene_path))

ground_truth = np.load(
    str(depth_path)
).astype(np.float32)

if scene is None:
    raise RuntimeError(
        "Could not load hospital corridor image."
    )

# Ensure correct image dimensions
if scene.shape[:2] != (HEIGHT, WIDTH):
    scene = cv2.resize(
        scene,
        (WIDTH, HEIGHT)
    )

# Ensure correct depth dimensions
if ground_truth.shape != (HEIGHT, WIDTH):
    ground_truth = cv2.resize(
        ground_truth,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

print("\nStage 1 data loaded successfully.")


# ============================================================
# 3. CREATE LEFT CAMERA IMAGE
# ============================================================

gray = cv2.cvtColor(
    scene,
    cv2.COLOR_BGR2GRAY
)

# Controlled fine texture.
#
# This gives the stereo matcher visual information
# while maintaining the hospital appearance.

rng = np.random.default_rng(42)

texture = rng.normal(
    loc=0.0,
    scale=7.0,
    size=(HEIGHT, WIDTH)
).astype(np.float32)

textured_left = (
    gray.astype(np.float32) +
    texture
)

# Mild smoothing to avoid unrealistic pixel noise
textured_left = cv2.GaussianBlur(
    textured_left,
    (3, 3),
    0
)

left_img = np.clip(
    textured_left,
    0,
    255
).astype(np.uint8)


# ============================================================
# 4. CALCULATE THEORETICAL DISPARITY
# ============================================================

# Stereo depth equation:
#
#                 f * B
#        Z = ----------------
#                  d
#
# Therefore:
#
#                 f * B
#        d = ----------------
#                   Z

disparity_gt = (
    FOCAL_LENGTH * BASELINE
) / np.maximum(
    ground_truth,
    0.1
)


# ============================================================
# 5. GENERATE RIGHT STEREO IMAGE
# ============================================================

# For a rectified stereo camera:
#
# x_right = x_left - disparity
#
# Therefore, when generating the right image:
#
# x_left = x_right + disparity

x_coords, y_coords = np.meshgrid(
    np.arange(
        WIDTH,
        dtype=np.float32
    ),
    np.arange(
        HEIGHT,
        dtype=np.float32
    )
)

map_x = (
    x_coords +
    disparity_gt
)

map_y = y_coords

right_img = cv2.remap(
    left_img,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)


# ============================================================
# 6. SIMULATE SMALL CAMERA ILLUMINATION DIFFERENCE
# ============================================================

right_img = np.clip(
    right_img.astype(np.float32) * 0.97 + 3,
    0,
    255
).astype(np.uint8)


# ============================================================
# 7. SAVE STEREO DATA
# ============================================================

cv2.imwrite(
    str(DATA_DIR / "stereo_left.png"),
    left_img
)

cv2.imwrite(
    str(DATA_DIR / "stereo_right.png"),
    right_img
)

np.save(
    str(DATA_DIR / "ground_truth_disparity.npy"),
    disparity_gt.astype(np.float32)
)

print("\nStereo images generated successfully.")


# ============================================================
# 8. CONVENTIONAL STEREO MATCHING
# ============================================================

print("\nRunning conventional StereoSGBM...")

# Disparity range must be divisible by 16.
num_disparities = 16 * 12

block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=0,

    numDisparities=num_disparities,

    blockSize=block_size,

    P1=8 * block_size * block_size,

    P2=32 * block_size * block_size,

    disp12MaxDiff=1,

    uniquenessRatio=8,

    speckleWindowSize=80,

    speckleRange=2,

    preFilterCap=63,

    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


# Calculate disparity
disparity_raw = stereo.compute(
    left_img,
    right_img
).astype(np.float32) / 16.0


# ============================================================
# 9. CONVERT DISPARITY TO DEPTH
# ============================================================

# Z = fB / d

valid_disp = (
    disparity_raw > 0.5
)

depth_stereo = np.zeros_like(
    ground_truth
)

depth_stereo[valid_disp] = (
    FOCAL_LENGTH * BASELINE
) / disparity_raw[valid_disp]


# ============================================================
# 10. EVALUATION MASK
# ============================================================

evaluation_mask = (
    valid_disp
    &
    np.isfinite(depth_stereo)
    &
    (ground_truth > 0.5)
    &
    (ground_truth < 10.0)
)

gt_values = ground_truth[
    evaluation_mask
]

pred_values = depth_stereo[
    evaluation_mask
]


if len(gt_values) == 0:

    raise RuntimeError(
        "No valid stereo depth pixels were produced."
    )


# ============================================================
# 11. CALCULATE ERROR METRICS
# ============================================================

absolute_error = np.abs(
    pred_values -
    gt_values
)


# Mean Absolute Error
mae = np.mean(
    absolute_error
)


# Root Mean Square Error
rmse = np.sqrt(
    np.mean(
        (pred_values - gt_values) ** 2
    )
)


# Relative depth error
relative_error = np.mean(
    absolute_error /
    np.maximum(
        gt_values,
        1e-6
    )
)


# Percentage within 10 cm
accuracy_10 = np.mean(
    absolute_error <= 0.10
) * 100.0


# Percentage within 25 cm
accuracy_25 = np.mean(
    absolute_error <= 0.25
) * 100.0


# Percentage of image with valid depth
valid_percentage = (
    len(gt_values) /
    ground_truth.size
) * 100.0


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("CONVENTIONAL STEREO RESULTS")
print("=" * 65)

print(
    f"Valid depth coverage : "
    f"{valid_percentage:.2f}%"
)

print(
    f"MAE                  : "
    f"{mae:.4f} m"
)

print(
    f"RMSE                 : "
    f"{rmse:.4f} m"
)

print(
    f"Relative error       : "
    f"{relative_error:.4f}"
)

print(
    f"Accuracy <= 0.10 m   : "
    f"{accuracy_10:.2f}%"
)

print(
    f"Accuracy <= 0.25 m   : "
    f"{accuracy_25:.2f}%"
)


# ============================================================
# 13. SAVE NUMERICAL RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "stage2_baseline_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARE-Depth Stage 2 - "
        "Conventional Stereo Baseline\n"
    )

    f.write("=" * 60 + "\n")

    f.write(
        f"Image width: {WIDTH}\n"
    )

    f.write(
        f"Image height: {HEIGHT}\n"
    )

    f.write(
        f"Focal length: "
        f"{FOCAL_LENGTH:.2f} pixels\n"
    )

    f.write(
        f"Baseline: "
        f"{BASELINE:.4f} m\n"
    )

    f.write(
        f"Valid depth coverage: "
        f"{valid_percentage:.6f}%\n"
    )

    f.write(
        f"MAE: "
        f"{mae:.6f} m\n"
    )

    f.write(
        f"RMSE: "
        f"{rmse:.6f} m\n"
    )

    f.write(
        f"Relative error: "
        f"{relative_error:.6f}\n"
    )

    f.write(
        f"Accuracy <= 0.10 m: "
        f"{accuracy_10:.6f}%\n"
    )

    f.write(
        f"Accuracy <= 0.25 m: "
        f"{accuracy_25:.6f}%\n"
    )


print("\nResults saved to:")
print(results_file)


# ============================================================
# 14. SAVE RAW NUMERICAL ARRAYS
# ============================================================

np.save(
    str(RESULTS_DIR / "conventional_disparity.npy"),
    disparity_raw
)

np.save(
    str(RESULTS_DIR / "conventional_depth.npy"),
    depth_stereo
)

np.save(
    str(RESULTS_DIR / "conventional_error.npy"),
    np.abs(
        depth_stereo -
        ground_truth
    )
)


# ============================================================
# 15. FIGURE 1 - LEFT AND RIGHT STEREO IMAGES
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    left_img,
    cmap="gray"
)

plt.title(
    "Left Stereo Image"
)

plt.axis("off")


plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    right_img,
    cmap="gray"
)

plt.title(
    "Right Stereo Image"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    str(
        FIGURES_DIR /
        "stereo_pair.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 16. FIGURE 2 - GROUND-TRUTH DISPARITY
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.imshow(
    disparity_gt,
    cmap="plasma"
)

plt.colorbar(
    label="Disparity (pixels)"
)

plt.title(
    "Ground-Truth Stereo Disparity"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    str(
        FIGURES_DIR /
        "ground_truth_disparity.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 17. FIGURE 3 - CONVENTIONAL DISPARITY
# ============================================================

display_disparity = (
    disparity_raw.copy()
)

display_disparity[
    display_disparity < 0
] = 0


plt.figure(
    figsize=(9, 6)
)

plt.imshow(
    display_disparity,
    cmap="plasma"
)

plt.colorbar(
    label="Estimated Disparity (pixels)"
)

plt.title(
    "Conventional StereoSGBM Disparity"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    str(
        FIGURES_DIR /
        "conventional_disparity.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 18. FIGURE 4 - CONVENTIONAL DEPTH
# ============================================================

display_depth = (
    depth_stereo.copy()
)

display_depth[
    ~valid_disp
] = np.nan


plt.figure(
    figsize=(9, 6)
)

plt.imshow(
    display_depth,
    cmap="viridis"
)

plt.colorbar(
    label="Estimated Depth (m)"
)

plt.title(
    "Conventional Stereo Depth"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    str(
        FIGURES_DIR /
        "conventional_depth.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 19. FIGURE 5 - ABSOLUTE DEPTH ERROR
# ============================================================

error_map = np.abs(
    depth_stereo -
    ground_truth
)

error_map[
    ~valid_disp
] = np.nan


plt.figure(
    figsize=(9, 6)
)

plt.imshow(
    error_map,
    cmap="inferno",
    vmin=0,
    vmax=2.0
)

plt.colorbar(
    label="Absolute Depth Error (m)"
)

plt.title(
    "Conventional Stereo Absolute Error"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    str(
        FIGURES_DIR /
        "conventional_error_map.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 20. FINISH
# ============================================================

print("\n")
print("=" * 65)
print("STAGE 2 COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")

print(
    "  data/stereo_left.png"
)

print(
    "  data/stereo_right.png"
)

print(
    "  data/ground_truth_disparity.npy"
)

print(
    "  results/stage2_baseline_results.txt"
)

print(
    "  results/conventional_disparity.npy"
)

print(
    "  results/conventional_depth.npy"
)

print(
    "  results/conventional_error.npy"
)

print("\nStage 2 is ready for CARE-Depth development.")