
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "svm_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

results_path = os.path.join(
    RESULTS_DIR,
    "svm_summary.csv"
)

results = pd.read_csv(results_path)

# Sort models by mean test accuracy
results = results.sort_values(
    by="Mean_Test_Accuracy",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(results.to_string(index=False))


table_path = os.path.join(
    RESULTS_DIR,
    "final_comparison.csv"
)

results.to_csv(table_path, index=False)

# Plot mean test accuracy
plt.figure(figsize=(10, 6))

bars = plt.bar(
    results["Model"],
    results["Mean Test Accuracy (%)"]
)

plt.title(
    "Mean SVM Test Accuracy Using Different Feature Extractors"
)
plt.xlabel("Feature Extraction Model")
plt.ylabel("Mean Test Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)

# Add accuracy labels above each bar
for bar, accuracy in zip(
    bars,
    results["Mean Test Accuracy (%)"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plot_path = os.path.join(
    RESULTS_DIR,
    "accuracy_comparison.png"
)

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved comparison table:", table_path)
print("Saved accuracy chart:", plot_path)
