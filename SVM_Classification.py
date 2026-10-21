import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
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

RANDOM_STATES = [42, 100, 2026]
TEST_SIZE = 0.20

def train_and_evaluate(model_name, run_number, random_state):
    
    print(f"Model: {model_name} | Run: {run_number} | Random State: {random_state}")
    

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

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=random_state
    )

    train_indices, val_indices = next(
        splitter.split(X_train, y_train)
    )

    X_train_run = X_train[train_indices]
    y_train_run = y_train[train_indices]

    X_val_run = X_train[val_indices]
    y_val_run = y_train[val_indices]

    print("SVM training features:", X_train_run.shape)
    print("Validation features:", X_val_run.shape)


    # Create a fresh SVM pipeline for every run
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    start_time = time.time()

    svm_pipeline.fit(X_train_run, y_train_run)

    training_time = time.time() - start_time

    val_pred = svm_pipeline.predict(X_val_run)
    val_accuracy = accuracy_score(y_val_run, val_pred)

    test_pred = svm_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")
    print(f"Training time: {training_time:.2f} seconds")

    report = classification_report(
        y_test,
        test_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    model_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_run{run_number}_svm.joblib"
    )

    joblib.dump(svm_pipeline, model_path)

    
    report_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_run{run_number}_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Feature extractor: {model_name}\n")
        file.write(f"Run: {run_number}\n")
        file.write(f"Random state: {random_state}\n")
        file.write(f"Validation accuracy: {val_accuracy:.4f}\n")
        file.write(f"Test accuracy: {test_accuracy:.4f}\n")
        file.write(f"Training time: {training_time:.2f} seconds\n\n")
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(confusion_matrix(y_test, test_pred)))

    print("\nSaved model:", model_path)
    print("Saved report:", report_path)

    return {
        "Model": model_name,
        "Run": run_number,
        "Random State": random_state,
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
        "Test Accuracy (%)": test_accuracy * 100,
        "Training Time (seconds)": training_time
    }


def main():

    results = []

   
    for model_name in MODEL_NAMES:

        for run_number, random_state in enumerate(
            RANDOM_STATES, start=1
        ):

            result = train_and_evaluate(
                model_name,
                run_number,
                random_state
            )

            results.append(result)

   
    results_df = pd.DataFrame(results)

    results_path = os.path.join(
        OUTPUT_DIR,
        "svm_all_experiments.csv"
    )

    results_df.to_csv(results_path, index=False)

    print("\n\nALL NINE EXPERIMENTS")
    print(results_df.to_string(index=False))

    
    summary_df = results_df.groupby("Model").agg(
        Mean_Validation_Accuracy=(
            "Validation Accuracy", "mean"
        ),
        Std_Validation_Accuracy=(
            "Validation Accuracy", "std"
        ),
        Mean_Test_Accuracy=(
            "Test Accuracy", "mean"
        ),
        Std_Test_Accuracy=(
            "Test Accuracy", "std"
        ),
        Mean_Training_Time=(
            "Training Time (seconds)", "mean"
        )
    ).reset_index()

    summary_df["Mean Test Accuracy (%)"] = (
        summary_df["Mean_Test_Accuracy"] * 100
    )

    summary_df["Std Test Accuracy (percentage points)"] = (
        summary_df["Std_Test_Accuracy"] * 100
    )

    summary_df = summary_df.sort_values(
        by="Mean_Test_Accuracy",
        ascending=False
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "svm_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("\n\nFINAL COMPARISON: MEAN RESULTS")
    print(summary_df.to_string(index=False))

    print("\nAll experiments saved to:", results_path)
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()






