
import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

FEATURES_DIR = "features"
OUTPUT_DIR = "svm_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAMES = [
    "ResNet50",
    "MobileNetV2",
    "EfficientNetB0"
]


def train_and_evaluate(model_name):
    print(f"\n{'=' * 50}")
    print(f"SVM using {model_name} features")
    print("=" * 50)

    # Load extracted features
    file_path = os.path.join(
        FEATURES_DIR,
        f"{model_name}_features.npz"
    )

    data = np.load(file_path, allow_pickle=False)

    X_train = data["X_train"]
    y_train = data["y_train"].ravel()
    X_test = data["X_test"]
    y_test = data["y_test"].ravel()
    class_names = data["class_names"].tolist()

    print("Training features:", X_train.shape)
    print("Test features:", X_test.shape)

    # Standardize features, then train SVM
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    start_time = time.time()

    svm_pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Predict test labels
    y_pred = svm_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Accuracy (%): {accuracy * 100:.2f}%")
    print(f"Training time: {training_time:.2f} seconds")

    # Detailed evaluation
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save trained SVM pipeline
    model_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_svm.joblib"
    )
    joblib.dump(svm_pipeline, model_path)

    # Save detailed report
    report_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_report.txt"
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Test accuracy: {accuracy:.4f}\n")
        file.write(f"Training time: {training_time:.2f} seconds\n\n")
        file.write(report)

    print("\nSaved SVM model:", model_path)
    print("Saved report:", report_path)

    return {
        "Model": model_name,
        "Test Accuracy": accuracy,
        "Test Accuracy (%)": accuracy * 100,
        "Training Time (seconds)": training_time
    }


def main():
    results = []

    for model_name in MODEL_NAMES:
        result = train_and_evaluate(model_name)
        results.append(result)

    # Compare all three experiments
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Test Accuracy",
        ascending=False
    )

    print("\n\nFINAL SVM COMPARISON")
    print(results_df.to_string(index=False))

    results_path = os.path.join(
        OUTPUT_DIR,
        "svm_comparison.csv"
    )
    results_df.to_csv(results_path, index=False)

    print("\nComparison saved to:", results_path)


if __name__ == "__main__":
    main()
