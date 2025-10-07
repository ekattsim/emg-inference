import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Load your Keras model
keras_model = load_model("../AI_Models/cnn_flexion_Aiden.h5", compile=False)

# Save it as a SavedModel
SAVED_MODEL_DIR = "./models/native_saved_model"
keras_model.save(SAVED_MODEL_DIR)

# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=SAVED_MODEL_DIR,
   precision_mode=trt.TrtPrecisionMode.FP16
)

# Convert the model into TRT compatible segments
converter.convert()
converter.summary()

OUTPUT_SAVED_MODEL_DIR="./models/tftrt_saved_model"
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)
