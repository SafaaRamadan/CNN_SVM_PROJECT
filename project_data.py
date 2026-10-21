from dataset import load_cifar10_dataset
from sklearn.model_selection import train_test_split
from numpy import unique
import numpy as np


def prepare_data(test_size=0.2, random_state=42):
    (x_train, y_train), (x_test, y_test), class_names = load_cifar10_dataset()

    y_train = y_train.flatten()
    y_test = y_test.flatten()


    Training_subset_size = 5000
    Random_state = 42
    x_subset, _, y_subset, _ = train_test_split(
        x_train, y_train, train_size=Training_subset_size,stratify=y_train, random_state=Random_state
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_subset, y_subset, test_size=test_size, stratify=y_subset, random_state=Random_state
    )

    print(len(class_names), "classes:", class_names)
    print("Training images:", len(x_train))
    print("Validation images:", len(x_val))
    print("Testing images:", len(x_test))
    print("Training image shape:", x_train.shape)
    print("Validation image shape:", x_val.shape)
    print("Testing image shape:", x_test.shape)
    print("num of classes:", len(np.unique(y_train)))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names

if __name__ == "__main__":
    prepare_data()