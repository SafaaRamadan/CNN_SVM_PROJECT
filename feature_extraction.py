import os
import numpy as np
import tensorflow as tf
from project_data import prepare_data


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_STATE = 42
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(RANDOM_STATE)

(x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = (prepare_data())


# x_train contains 4000 images and x_val contains 1000
x_svm_train = np.concatenate([x_train, x_val], axis=0)
y_svm_train = np.concatenate([y_train, y_val], axis=0)

# The test set stays separate
print("SVM training images:", x_svm_train.shape)
print("Test images:", x_test.shape)


model_names = [
    "ResNet50",
    "MobileNetV2",
    "EfficientNetB0",
]

preprocess_functions = {
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
}


def prepare_images(images, preprocess_fn):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.batch(BATCH_SIZE)

    def preprocess_batch(batch):
        batch = tf.cast(batch, tf.float32)
        batch = tf.image.resize(batch, IMAGE_SIZE)
        batch = preprocess_fn(batch)
        return batch

    dataset = dataset.map(
        preprocess_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def extract_features(model_name):
    print(f"\nExtracting features using {model_name}...")

    model_path = os.path.join(
        "saved_models",
        "transfer_learning",
        f"{model_name}.keras"
    )

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Get the feature vector before Dropout and the final classifier
    feature_layer = model.get_layer("global_average_pooling2d")
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=feature_layer.output
    )

    preprocess_fn = preprocess_functions[model_name]

    # Prepare training and test images
    train_dataset = prepare_images(x_svm_train, preprocess_fn)
    test_dataset = prepare_images(x_test, preprocess_fn)

    # Extract features
    train_features = feature_extractor.predict(
        train_dataset,
        verbose=1
    )

    test_features = feature_extractor.predict(
        test_dataset,
        verbose=1
    )

    print("Training features:", train_features.shape)
    print("Test features:", test_features.shape)

   
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_features.npz"
    )

    np.savez_compressed(
        output_path,
        X_train=train_features,
        y_train=y_svm_train,
        X_test=test_features,
        y_test=y_test,
        class_names=np.array(class_names)
    )

    print("Saved features to:", output_path)

    del model, feature_extractor
    tf.keras.backend.clear_session()


def main():
    for model_name in model_names:
        extract_features(model_name)

    print("\nFeature extraction completed!")


if __name__ == "__main__":
    main()
