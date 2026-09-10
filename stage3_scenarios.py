import os
import cv2
import numpy as np

# ============================================================
# CARE-Depth - Stage 3
# Hospital Challenge Scenario Generator
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 64)
print("CARE-Depth Stage 3 - Hospital Challenge Scenarios")
print("=" * 64)

# ------------------------------------------------------------
# Load Stage 1 / Stage 2 data
# ------------------------------------------------------------

scene_path = os.path.join(DATA_DIR, "hospital_corridor.png")
depth_path = os.path.join(DATA_DIR, "ground_truth_depth.npy")

scene = cv2.imread(scene_path)

if scene is None:
    raise FileNotFoundError(
        "hospital_corridor.png was not found in the data folder."
    )

ground_truth = np.load(depth_path)

print("\nStage 1 data loaded successfully.")
print(f"Image size: {scene.shape[1]} x {scene.shape[0]}")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def save_scenario(name, image):
    path = os.path.join(DATA_DIR, f"scenario_{name}.png")
    cv2.imwrite(path, image)
    print(f"Generated: {name}")


def add_low_texture(image):
    """Reduce texture information in selected wall regions."""
    result = image.copy()

    h, w = result.shape[:2]

    # Left wall region
    left_region = result[int(0.25*h):int(0.75*h), :int(0.32*w)]
    blurred_left = cv2.GaussianBlur(left_region, (31, 31), 0)
    result[int(0.25*h):int(0.75*h), :int(0.32*w)] = blurred_left

    # Right wall region
    right_region = result[int(0.25*h):int(0.75*h), int(0.68*w):]
    blurred_right = cv2.GaussianBlur(right_region, (31, 31), 0)
    result[int(0.25*h):int(0.75*h), int(0.68*w):] = blurred_right

    return result


def add_illumination_change(image):
    """Simulate uneven illumination inside a hospital corridor."""
    result = image.astype(np.float32)

    h, w = result.shape[:2]

    # Horizontal illumination gradient
    gradient = np.linspace(0.65, 1.25, w)
    illumination = np.tile(gradient, (h, 1))

    for c in range(3):
        result[:, :, c] *= illumination

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def add_sensor_noise(image):
    """Simulate camera sensor noise."""
    result = image.astype(np.float32)

    noise = np.random.normal(
        loc=0,
        scale=18,
        size=result.shape
    )

    result += noise

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def add_moving_pedestrian(image):
    """Simulate a pedestrian moving through the camera field."""
    result = image.copy()

    h, w = result.shape[:2]

    # Create a synthetic pedestrian silhouette
    center_x = int(0.58 * w)
    center_y = int(0.57 * h)

    # Head
    cv2.circle(
        result,
        (center_x, center_y - 55),
        24,
        (90, 90, 90),
        -1
    )

    # Body
    cv2.rectangle(
        result,
        (center_x - 25, center_y - 30),
        (center_x + 25, center_y + 65),
        (75, 75, 75),
        -1
    )

    # Legs
    cv2.line(
        result,
        (center_x - 10, center_y + 65),
        (center_x - 30, center_y + 125),
        (65, 65, 65),
        18
    )

    cv2.line(
        result,
        (center_x + 10, center_y + 65),
        (center_x + 35, center_y + 125),
        (65, 65, 65),
        18
    )

    return result


# ------------------------------------------------------------
# Generate scenarios
# ------------------------------------------------------------

np.random.seed(42)

scenarios = {}

# 1. Normal
scenarios["normal"] = scene.copy()

# 2. Low texture
scenarios["low_texture"] = add_low_texture(scene)

# 3. Illumination change
scenarios["illumination_change"] = add_illumination_change(scene)

# 4. Sensor noise
scenarios["sensor_noise"] = add_sensor_noise(scene)

# 5. Moving pedestrian
scenarios["moving_pedestrian"] = add_moving_pedestrian(scene)

# 6. Combined difficult hospital condition
combined = add_low_texture(scene)
combined = add_illumination_change(combined)
combined = add_sensor_noise(combined)
combined = add_moving_pedestrian(combined)

scenarios["combined"] = combined

# ------------------------------------------------------------
# Save scenarios
# ------------------------------------------------------------

for name, image in scenarios.items():
    save_scenario(name, image)

# Save ground truth for reference
np.save(
    os.path.join(DATA_DIR, "stage3_ground_truth_depth.npy"),
    ground_truth
)

# ------------------------------------------------------------
# Save experiment description
# ------------------------------------------------------------

results_file = os.path.join(
    RESULTS_DIR,
    "stage3_scenarios.txt"
)

with open(results_file, "w", encoding="utf-8") as f:

    f.write("CARE-Depth Stage 3\n")
    f.write("Hospital Challenge Scenario Generation\n")
    f.write("=" * 60 + "\n\n")

    f.write("Synthetic scenarios generated:\n")
    f.write("1. Normal corridor\n")
    f.write("2. Low-texture walls\n")
    f.write("3. Illumination change\n")
    f.write("4. Sensor noise\n")
    f.write("5. Moving pedestrian\n")
    f.write("6. Combined difficult condition\n\n")

    f.write(
        "These scenarios are controlled synthetic perturbations "
        "designed to evaluate depth-estimation robustness.\n"
    )

print("\n" + "=" * 64)
print("STAGE 3 COMPLETED SUCCESSFULLY")
print("=" * 64)

print("\nScenarios generated:")
for name in scenarios:
    print(f"  - {name}")

print("\nFiles saved in:")
print(DATA_DIR)

print("\nNext step: Stage 4 - CARE-Depth reliability and temporal fusion.")