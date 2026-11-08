import tensorflow as tf
import numpy as np
from dataset import load_cifar10_dataset
from models import display_models

(x_train, y_train), (x_test, y_test), class_names = load_cifar10_dataset()


x_sample = x_test[:100]


preprocessing_functions = {
    "DenseNet121": tf.keras.applications.densenet.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
    "InceptionV3": tf.keras.applications.inception_v3.preprocess_input
}

for name, model in display_models.items():
    print(f"Testing {name}")
  

    if name == "InceptionV3":
        resized_images = tf.image.resize(x_test[:100], (299, 299)).numpy()
    else:
        resized_images = tf.image.resize(x_test[:100], (224, 224)).numpy()

    preprocess = preprocessing_functions[name]
    images = preprocess(resized_images.copy())
    predictions = model.predict(images, batch_size=8, verbose=0)

    if name == "DenseNet121":
        decoded = tf.keras.applications.densenet.decode_predictions(
        predictions, top=3
    )
    elif name == "VGG16":
        decoded = tf.keras.applications.vgg16.decode_predictions(
        predictions, top=3
    )
    elif name == "InceptionV3":
        decoded = tf.keras.applications.inception_v3.decode_predictions(
        predictions, top=3
    )

    for i in range(5):
        true_label = int(y_test[i, 0])

        print(f"\nImage {i}")
        print(f"True CIFAR-10 class: {class_names[true_label]}")
        print("Top 3 ImageNet predictions:")

        for class_id, class_name, score in decoded[i]:
            print(class_id, class_name, f"{score:.2%}")