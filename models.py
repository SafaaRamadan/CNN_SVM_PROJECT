import tensorflow as tf

IMAGE_SIZE = (224 , 224)

print("Loading pretrained ResNet50 model")
resnet_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=True,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

print("Loading pretrained MobileNetV2 model")
mobilenet_model = tf.keras.applications.MobileNetV2(    
    weights="imagenet",
    include_top=True,   
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

print("Loading pretrained EfficientNetB0 model")
efficientnet_model = tf.keras.applications.EfficientNetB0(  
    weights="imagenet",
    include_top=True,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

display_models = {
    "ResNet50": resnet_model,   
    "MobileNetV2": mobilenet_model,
    "EfficientNetB0": efficientnet_model
}
def display_model_info():
    for name, model in display_models.items():
        print(f"Model: {name}")
        print("Input_shape: ", model.input_shape)
        print("Output_shape: ", model.output_shape)
        print("Number of parameters:", model.count_params())

if __name__ == "__main__":
    display_model_info()