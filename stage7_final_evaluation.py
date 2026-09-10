import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CARE-Depth Stage 7
# Final Evaluation and Analysis
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


print("=" * 70)
print("CARE-Depth Stage 7 - Final Evaluation and Analysis")
print("=" * 70)


# ------------------------------------------------------------
# Load Stage 6 results
# ------------------------------------------------------------

csv_path = os.path.join(
    RESULTS_DIR,
    "stage6_care_depth_v2_results.csv"
)

if not os.path.exists(csv_path):
    print("\nERROR: Stage 6 results file was not found.")
    print(csv_path)
    raise SystemExit


df = pd.read_csv(csv_path)

print("\nStage 6 results loaded successfully.")
print("Number of scenarios:", len(df))


# ------------------------------------------------------------
# Calculate changes
# ------------------------------------------------------------

df["MAE_change_m"] = (
    df["baseline_MAE_m"]
    - df["CARE_MAE_m"]
)

df["RMSE_change_m"] = (
    df["baseline_RMSE_m"]
    - df["CARE_RMSE_m"]
)

df["accuracy_10cm_change_%"] = (
    df["CARE_accuracy_10cm_%"]
    - df["baseline_accuracy_10cm_%"]
)

df["accuracy_25cm_change_%"] = (
    df["CARE_accuracy_25cm_%"]
    - df["baseline_accuracy_25cm_%"]
)

df["coverage_change_%"] = (
    df["CARE_coverage_%"]
    - df["baseline_coverage_%"]
)


# ------------------------------------------------------------
# Overall averages
# ------------------------------------------------------------

overall = {
    "baseline_MAE_m":
        df["baseline_MAE_m"].mean(),

    "CARE_MAE_m":
        df["CARE_MAE_m"].mean(),

    "baseline_RMSE_m":
        df["baseline_RMSE_m"].mean(),

    "CARE_RMSE_m":
        df["CARE_RMSE_m"].mean(),

    "baseline_accuracy_10cm_%":
        df["baseline_accuracy_10cm_%"].mean(),

    "CARE_accuracy_10cm_%":
        df["CARE_accuracy_10cm_%"].mean(),

    "baseline_accuracy_25cm_%":
        df["baseline_accuracy_25cm_%"].mean(),

    "CARE_accuracy_25cm_%":
        df["CARE_accuracy_25cm_%"].mean(),

    "baseline_coverage_%":
        df["baseline_coverage_%"].mean(),

    "CARE_coverage_%":
        df["CARE_coverage_%"].mean(),
}


overall_mae_change = (
    overall["baseline_MAE_m"]
    - overall["CARE_MAE_m"]
)

overall_rmse_change = (
    overall["baseline_RMSE_m"]
    - overall["CARE_RMSE_m"]
)

overall_acc10_change = (
    overall["CARE_accuracy_10cm_%"]
    - overall["baseline_accuracy_10cm_%"]
)

overall_acc25_change = (
    overall["CARE_accuracy_25cm_%"]
    - overall["baseline_accuracy_25cm_%"]
)

overall_coverage_change = (
    overall["CARE_coverage_%"]
    - overall["baseline_coverage_%"]
)


# ------------------------------------------------------------
# Print final analysis
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("OVERALL RESULTS")
print("-" * 70)

print(
    f"Average Baseline MAE : "
    f"{overall['baseline_MAE_m']:.4f} m"
)

print(
    f"Average CARE MAE     : "
    f"{overall['CARE_MAE_m']:.4f} m"
)

print(
    f"Average MAE change   : "
    f"{overall_mae_change:+.4f} m"
)

print()

print(
    f"Average Baseline RMSE: "
    f"{overall['baseline_RMSE_m']:.4f} m"
)

print(
    f"Average CARE RMSE    : "
    f"{overall['CARE_RMSE_m']:.4f} m"
)

print(
    f"Average RMSE change  : "
    f"{overall_rmse_change:+.4f} m"
)

print()

print(
    f"Baseline Accuracy <= 10 cm: "
    f"{overall['baseline_accuracy_10cm_%']:.2f}%"
)

print(
    f"CARE Accuracy <= 10 cm    : "
    f"{overall['CARE_accuracy_10cm_%']:.2f}%"
)

print(
    f"Accuracy change            : "
    f"{overall_acc10_change:+.2f}%"
)

print()

print(
    f"Baseline Accuracy <= 25 cm: "
    f"{overall['baseline_accuracy_25cm_%']:.2f}%"
)

print(
    f"CARE Accuracy <= 25 cm    : "
    f"{overall['CARE_accuracy_25cm_%']:.2f}%"
)

print(
    f"Accuracy change            : "
    f"{overall_acc25_change:+.2f}%"
)

print()

print(
    f"Baseline coverage: "
    f"{overall['baseline_coverage_%']:.2f}%"
)

print(
    f"CARE coverage    : "
    f"{overall['CARE_coverage_%']:.2f}%"
)

print(
    f"Coverage change  : "
    f"{overall_coverage_change:+.2f}%"
)


# ------------------------------------------------------------
# Determine scenario-level outcome
# ------------------------------------------------------------

better_mae = np.sum(df["MAE_change_m"] > 0)
equal_mae = np.sum(df["MAE_change_m"] == 0)
worse_mae = np.sum(df["MAE_change_m"] < 0)

print("\n" + "-" * 70)
print("SCENARIO-LEVEL MAE ANALYSIS")
print("-" * 70)

print("Scenarios where CARE improves MAE :", better_mae)
print("Scenarios with equal MAE          :", equal_mae)
print("Scenarios where CARE worsens MAE  :", worse_mae)


# ------------------------------------------------------------
# Save final CSV
# ------------------------------------------------------------

final_csv = os.path.join(
    RESULTS_DIR,
    "stage7_final_evaluation.csv"
)

df.to_csv(
    final_csv,
    index=False
)


# ------------------------------------------------------------
# Save final text report
# ------------------------------------------------------------

report_path = os.path.join(
    RESULTS_DIR,
    "stage7_final_evaluation.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARE-Depth Stage 7 - Final Evaluation\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "The Stage 6 CARE-Depth V2 results were "
        "evaluated across six hospital challenge scenarios.\n\n"
    )

    f.write("OVERALL RESULTS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Average Baseline MAE : "
        f"{overall['baseline_MAE_m']:.6f} m\n"
    )

    f.write(
        f"Average CARE MAE     : "
        f"{overall['CARE_MAE_m']:.6f} m\n"
    )

    f.write(
        f"Average MAE change   : "
        f"{overall_mae_change:+.6f} m\n\n"
    )

    f.write(
        f"Average Baseline RMSE: "
        f"{overall['baseline_RMSE_m']:.6f} m\n"
    )

    f.write(
        f"Average CARE RMSE    : "
        f"{overall['CARE_RMSE_m']:.6f} m\n"
    )

    f.write(
        f"Average RMSE change  : "
        f"{overall_rmse_change:+.6f} m\n\n"
    )

    f.write(
        f"Baseline Accuracy <= 10 cm: "
        f"{overall['baseline_accuracy_10cm_%']:.4f}%\n"
    )

    f.write(
        f"CARE Accuracy <= 10 cm    : "
        f"{overall['CARE_accuracy_10cm_%']:.4f}%\n"
    )

    f.write(
        f"Accuracy change            : "
        f"{overall_acc10_change:+.4f}%\n\n"
    )

    f.write(
        f"Baseline Accuracy <= 25 cm: "
        f"{overall['baseline_accuracy_25cm_%']:.4f}%\n"
    )

    f.write(
        f"CARE Accuracy <= 25 cm    : "
        f"{overall['CARE_accuracy_25cm_%']:.4f}%\n"
    )

    f.write(
        f"Accuracy change            : "
        f"{overall_acc25_change:+.4f}%\n\n"
    )

    f.write(
        f"Baseline coverage: "
        f"{overall['baseline_coverage_%']:.4f}%\n"
    )

    f.write(
        f"CARE coverage    : "
        f"{overall['CARE_coverage_%']:.4f}%\n"
    )

    f.write(
        f"Coverage change  : "
        f"{overall_coverage_change:+.4f}%\n\n"
    )

    f.write("SCENARIO-LEVEL ANALYSIS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"CARE improved MAE in {better_mae} of "
        f"{len(df)} scenarios.\n"
    )

    f.write(
        f"CARE had equal MAE in {equal_mae} of "
        f"{len(df)} scenarios.\n"
    )

    f.write(
        f"CARE worsened MAE in {worse_mae} of "
        f"{len(df)} scenarios.\n\n"
    )

    f.write("DETAILED RESULTS\n")
    f.write("-" * 70 + "\n")

    f.write(
        df.to_string(index=False)
    )


# ------------------------------------------------------------
# Figure 1 - MAE comparison
# ------------------------------------------------------------

x = np.arange(len(df))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    df["baseline_MAE_m"],
    width,
    label="Conventional Stereo"
)

plt.bar(
    x + width / 2,
    df["CARE_MAE_m"],
    width,
    label="CARE-Depth V2"
)

plt.xticks(
    x,
    df["scenario"],
    rotation=30,
    ha="right"
)

plt.ylabel("MAE (m)")
plt.title("MAE Comparison Across Hospital Scenarios")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "stage7_MAE_comparison.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# Figure 2 - RMSE comparison
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    df["baseline_RMSE_m"],
    width,
    label="Conventional Stereo"
)

plt.bar(
    x + width / 2,
    df["CARE_RMSE_m"],
    width,
    label="CARE-Depth V2"
)

plt.xticks(
    x,
    df["scenario"],
    rotation=30,
    ha="right"
)

plt.ylabel("RMSE (m)")
plt.title("RMSE Comparison Across Hospital Scenarios")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "stage7_RMSE_comparison.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# Figure 3 - Accuracy comparison
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    df["scenario"],
    df["baseline_accuracy_10cm_%"],
    marker="o",
    label="Stereo <= 10 cm"
)

plt.plot(
    df["scenario"],
    df["CARE_accuracy_10cm_%"],
    marker="o",
    label="CARE V2 <= 10 cm"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylabel("Accuracy (%)")
plt.title("10 cm Depth Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "stage7_accuracy_10cm.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# Figure 4 - Coverage comparison
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    df["baseline_coverage_%"],
    width,
    label="Conventional Stereo"
)

plt.bar(
    x + width / 2,
    df["CARE_coverage_%"],
    width,
    label="CARE-Depth V2"
)

plt.xticks(
    x,
    df["scenario"],
    rotation=30,
    ha="right"
)

plt.ylabel("Coverage (%)")
plt.title("Valid Depth Coverage Across Hospital Scenarios")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "stage7_coverage_comparison.png"
    ),
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STAGE 7 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSaved:")
print(final_csv)
print(report_path)

print("\nFigures saved to:")
print(FIGURES_DIR)

print("\nStage 7 analysis is complete.")