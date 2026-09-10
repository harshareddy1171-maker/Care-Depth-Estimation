import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CARE-Depth: Synthetic Hospital Corridor Generator
# ============================================================

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Image dimensions
WIDTH = 640
HEIGHT = 480

# ------------------------------------------------------------
# 1. Create synthetic hospital corridor
# ------------------------------------------------------------

def create_hospital_scene():
    """
    Creates a synthetic hospital corridor with:
    - floor
    - walls
    - ceiling
    - medicine cart
    - hospital bed
    - person
    - low-texture wall regions
    """

    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Background wall
    image[:, :] = [235, 235, 235]

    # Corridor perspective boundaries
    vanishing_x = WIDTH // 2
    vanishing_y = 150

    # Floor
    floor_polygon = np.array([
        [0, HEIGHT],
        [WIDTH, HEIGHT],
        [vanishing_x + 80, vanishing_y],
        [vanishing_x - 80, vanishing_y]
    ], dtype=np.int32)

    cv2.fillPoly(image, [floor_polygon], (190, 190, 190))

    # Left wall
    left_wall = np.array([
        [0, 0],
        [vanishing_x - 80, vanishing_y],
        [vanishing_x - 100, HEIGHT],
        [0, HEIGHT]
    ], dtype=np.int32)

    cv2.fillPoly(image, [left_wall], (225, 225, 225))

    # Right wall
    right_wall = np.array([
        [WIDTH, 0],
        [vanishing_x + 80, vanishing_y],
        [vanishing_x + 100, HEIGHT],
        [WIDTH, HEIGHT]
    ], dtype=np.int32)

    cv2.fillPoly(image, [right_wall], (230, 230, 230))

    # Ceiling
    ceiling = np.array([
        [0, 0],
        [WIDTH, 0],
        [vanishing_x + 80, vanishing_y],
        [vanishing_x - 80, vanishing_y]
    ], dtype=np.int32)

    cv2.fillPoly(image, [ceiling], (245, 245, 245))

    # --------------------------------------------------------
    # Hospital environmental features
    # --------------------------------------------------------

    # Wall panels / low texture structure
    for x in [100, 200, 440, 540]:
        cv2.line(
            image,
            (x, 50),
            (WIDTH // 2 + int((x - WIDTH // 2) * 0.18), 145),
            (200, 200, 200),
            2
        )

    # Door on right side
    cv2.rectangle(image, (500, 110), (575, 290), (180, 180, 180), -1)
    cv2.rectangle(image, (510, 120), (565, 280), (210, 210, 210), 2)
    cv2.circle(image, (550, 205), 4, (80, 80, 80), -1)

    # Medicine cart
    cv2.rectangle(image, (120, 270), (210, 370), (150, 150, 150), -1)
    cv2.rectangle(image, (130, 285), (200, 315), (220, 220, 220), -1)
    cv2.rectangle(image, (130, 325), (200, 350), (210, 210, 210), -1)
    cv2.circle(image, (140, 375), 12, (60, 60, 60), -1)
    cv2.circle(image, (190, 375), 12, (60, 60, 60), -1)

    # Hospital bed
    cv2.rectangle(image, (380, 290), (520, 350), (205, 205, 205), -1)
    cv2.rectangle(image, (365, 275), (390, 350), (150, 150, 150), -1)
    cv2.rectangle(image, (400, 275), (500, 305), (235, 235, 235), -1)
    cv2.line(image, (395, 350), (395, 390), (70, 70, 70), 5)
    cv2.line(image, (500, 350), (500, 390), (70, 70, 70), 5)

    # Person / pedestrian
    cv2.circle(image, (315, 230), 22, (160, 160, 160), -1)
    cv2.rectangle(image, (295, 250), (335, 335), (130, 130, 130), -1)
    cv2.line(image, (305, 335), (300, 395), (80, 80, 80), 8)
    cv2.line(image, (325, 335), (335, 395), (80, 80, 80), 8)

    # Ceiling lights
    cv2.rectangle(image, (270, 35), (370, 48), (255, 255, 255), -1)
    cv2.rectangle(image, (285, 80), (355, 90), (255, 255, 255), -1)

    # Add subtle sensor noise
    noise = np.random.normal(0, 2.0, image.shape)
    noisy_image = image.astype(np.float32) + noise

    return np.clip(noisy_image, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# 2. Create ground-truth depth map
# ------------------------------------------------------------

def create_ground_truth_depth():
    """
    Creates a synthetic metric depth map in meters.

    The corridor depth increases toward the vanishing point.
    Objects are assigned known distances.
    """

    y, x = np.indices((HEIGHT, WIDTH))

    # Base corridor depth
    normalized_y = np.clip(
        (HEIGHT - y) / (HEIGHT - 150),
        0,
        1
    )

    depth = 2.0 + normalized_y * 8.0

    # Floor becomes farther toward horizon
    floor_mask = y > 150
    depth[floor_mask] = (
        1.8 + ((HEIGHT - y[floor_mask]) / (HEIGHT - 150)) * 8.2
    )

    # Medicine cart: approximately 3.0 m
    cart_mask = (
        (x >= 120) & (x <= 210) &
        (y >= 270) & (y <= 390)
    )
    depth[cart_mask] = 3.0

    # Person: approximately 2.2 m
    person_mask = (
        ((x - 315) ** 2 + (y - 230) ** 2 < 22 ** 2) |
        ((x >= 295) & (x <= 335) & (y >= 250) & (y <= 335))
    )
    depth[person_mask] = 2.2

    # Hospital bed: approximately 3.5 m
    bed_mask = (
        (x >= 365) & (x <= 520) &
        (y >= 275) & (y <= 390)
    )
    depth[bed_mask] = 3.5

    # Door: approximately 4.5 m
    door_mask = (
        (x >= 500) & (x <= 575) &
        (y >= 110) & (y <= 290)
    )
    depth[door_mask] = 4.5

    return depth.astype(np.float32)


# ------------------------------------------------------------
# 3. Save and display scene
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("CARE-Depth Synthetic Hospital Simulation")
    print("=" * 60)

    scene = create_hospital_scene()
    ground_truth = create_ground_truth_depth()

    # Save data
    cv2.imwrite(
        str(DATA_DIR / "hospital_corridor.png"),
        cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    )

    np.save(
        str(DATA_DIR / "ground_truth_depth.npy"),
        ground_truth
    )

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(scene)
    plt.title("Synthetic Hospital Corridor")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        str(FIGURES_DIR / "hospital_corridor.png"),
        dpi=200
    )
    plt.show()

    # Ground truth visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(ground_truth, cmap="viridis")
    plt.colorbar(label="Depth (m)")
    plt.title("Ground-Truth Depth Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        str(FIGURES_DIR / "ground_truth_depth.png"),
        dpi=200
    )
    plt.show()

    print("\nSimulation data generated successfully.")
    print(f"Scene saved to: {DATA_DIR / 'hospital_corridor.png'}")
    print(f"Depth saved to: {DATA_DIR / 'ground_truth_depth.npy'}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()