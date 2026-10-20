
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import (decode_predictions)
from dataset import load_cifar10_dataset
from models import display_models

(x_train, y_train), (x_test, y_test), class_names = load_cifar10_dataset()


x_sample = x_test[:100]
x_sample = tf.image.resize(x_sample, (224, 224))
x_sample = x_sample.numpy()

preprocessing_functions = {
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input
}

for name, model in display_models.items():

    print(f"\n{'=' * 50}")
    print(f"Testing {name}")
    print("=" * 50)

    preprocess = preprocessing_functions[name]
    images = preprocess(x_sample.copy())

    predictions = model.predict(images, verbose=0)
    decoded = decode_predictions(predictions, top=3)

    for i in range(5):
        true_label = int(y_test[i, 0])

        print(f"\nImage {i}")
        print(f"True CIFAR-10 class: {class_names[true_label]}")
        print("Top 3 ImageNet predictions:")

        for class_id, class_name, score in decoded[i]:
            print(class_id, class_name, f"{score:.2%}")