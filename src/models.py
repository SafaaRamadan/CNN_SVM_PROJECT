"""Registry of pretrained backbones: name -> (constructor, preprocess_fn, input_size)."""
import tensorflow as tf

K = tf.keras.applications

BACKBONES = {
    "DenseNet121": (K.DenseNet121, K.densenet.preprocess_input, (224, 224)),
    "VGG16": (K.VGG16, K.vgg16.preprocess_input, (224, 224)),
    "InceptionV3": (K.InceptionV3, K.inception_v3.preprocess_input, (299, 299)),
    "ResNet50": (K.ResNet50, K.resnet50.preprocess_input, (224, 224)),
    # EfficientNetV2 has preprocessing built into the model (preprocess_input is identity)
    "EfficientNetV2B0": (K.EfficientNetV2B0, K.efficientnet_v2.preprocess_input, (224, 224)),
}


def build_extractor(name):
    ctor, pre, size = BACKBONES[name]
    model = ctor(weights="imagenet", include_top=False, pooling="avg",
                 input_shape=(*size, 3))
    model.trainable = False
    return model, pre, size
