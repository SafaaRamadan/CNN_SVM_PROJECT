
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    return (x_train, y_train), (x_test, y_test), class_names


def show_dataset_info():
    (x_train, y_train), (x_test, y_test), class_names = (
        load_cifar10_dataset()
    )

    classes = np.unique(y_train)

    print("Number of classes:", len(classes))
    print("Class labels:", classes)
    print("Training images:", len(x_train))
    print("Testing images:", len(x_test))
    print("Training image shape:", x_train.shape)

    train_labels = y_train.flatten()
    test_labels = y_test.flatten()

    print("\n========== IMAGES PER CLASS ==========")

    for i, class_name in enumerate(class_names):
        train_count = np.sum(train_labels == i)
        test_count = np.sum(test_labels == i)

        print(
            f"{class_name:12s} | "
            f"Train: {train_count:5d} | "
            f"Test: {test_count:5d} | "
            f"Total: {train_count + test_count:5d}"
        )

    print("\nDisplaying sample training images...")

    plt.figure(figsize=(12, 6))

    for i in range(15):
        plt.subplot(3, 5, i + 1)

        image = x_train[i]
        label = int(y_train[i][0])

        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis("off")

    plt.suptitle("CIFAR-10 Training Samples", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_dataset_info()
