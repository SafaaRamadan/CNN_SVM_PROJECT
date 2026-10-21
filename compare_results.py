
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "svm_results"

results_path = os.path.join(
    RESULTS_DIR,
    "svm_comparison.csv"
)

# Load SVM comparison results
results = pd.read_csv(results_path)

# Sort by test accuracy
results = results.sort_values(
    by="Test Accuracy",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(results.to_string(index=False))

# Save a clean comparison table
table_path = os.path.join(
    RESULTS_DIR,
    "final_comparison.csv"
)
results.to_csv(table_path, index=False)

# Plot accuracy comparison
plt.figure(figsize=(10, 6))

bars = plt.bar(
    results["Model"],
    results["Test Accuracy (%)"]
)

plt.title("SVM Accuracy Using Different Feature Extractors")
plt.xlabel("Feature Extraction Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)

for bar, accuracy in zip(
    bars,
    results["Test Accuracy (%)"]
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

plot_path = os.path.join(
    RESULTS_DIR,
    "accuracy_comparison.png"
)

plt.savefig(plot_path, dpi=300)
plt.show()

print("\nSaved comparison table:", table_path)
print("Saved accuracy chart:", plot_path)
