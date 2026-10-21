
import os
import tensorflow as tf

from project_data import prepare_data



IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 10
RANDOM_STATE = 42

OUTPUT_DIR = "saved_models/transfer_learning"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(RANDOM_STATE)



(x_train, y_train), (x_val, y_val), _, class_names = prepare_data()


model_configs = {
    "ResNet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
    ),
    "MobileNetV2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "EfficientNetB0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
}


def prepare_tf_dataset(images, labels, preprocess_fn, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(images),
            seed=RANDOM_STATE
        )

    dataset = dataset.batch(BATCH_SIZE)

    def preprocess_batch(images, labels):
        images = tf.cast(images, tf.float32)

        images = tf.image.resize(
            images,
            IMAGE_SIZE
        )

        images = preprocess_fn(images)

        return images, labels

    dataset = dataset.map(
        preprocess_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def build_transfer_model(base_model_fn):

    base_model = base_model_fn(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3)
    )

    features = base_model(
        inputs,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="cifar10_classifier"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    results = {}

    for model_name, (base_model_fn, preprocess_fn) in model_configs.items():

        print(f"Transfer Learning: {model_name}")
      

        tf.keras.backend.clear_session()

        train_dataset = prepare_tf_dataset(
            x_train,
            y_train,
            preprocess_fn,
            training=True
        )

        val_dataset = prepare_tf_dataset(
            x_val,
            y_val,
            preprocess_fn
        )

        model = build_transfer_model(base_model_fn)

        model.summary()

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            verbose=1
        )

        val_loss, val_accuracy = model.evaluate(
            val_dataset,
            verbose=0
        )

        model_path = os.path.join(
            OUTPUT_DIR,
            f"{model_name}.keras"
        )

        model.save(model_path)

        results[model_name] = {
            "validation_accuracy": val_accuracy,
            "validation_loss": val_loss,
            "model_path": model_path
        }

        print(f"\n{model_name} results:")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Saved model: {model_path}")

        del model, history, train_dataset, val_dataset


    print("TRANSFER LEARNING SUMMARY")
    for name, result in results.items():
        print(
            f"{name:15s} | "
            f"Validation Accuracy: "
            f"{result['validation_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
