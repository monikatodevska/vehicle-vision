import helper_model1
import os
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

modelsPath = r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva\new_277'
modelPathOrig=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva\model.json'
modelPathOrigw=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva\model.h5'
# model = helper_model1.construct_model_anchorless_detector_Pedestrians_anchorless(input_shape=(281, 469, 1))

# model = helper_model1.construct_model_anchorless_detector_skip_v1_custom_loss(input_shape=(277, 512, 1))
model = helper_model1.construct_model_anchorless_detector_skip_v1_custom_loss(input_shape=(277, 512, 1))
os.makedirs(modelsPath,exist_ok=True)
print(model.summary())

with open(modelPathOrig, "r") as f:
    model_json = f.read()

old_model = model_from_json(model_json)
old_model.load_weights(modelPathOrigw)
print("Old model loaded successfully")

# model_json = model.to_json()  # serialize model architecture to JSON
# with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
#     json_file.write(model_json)
# model.load_weights(os.path.join(modelPathOrig,'model.h5'))
# model.save_weights(os.path.join(modelsPath, 'model.h5'))
# print(model.summary())
#⃣ Copy weights layer by layer, skip Lambda / mismatched shapes
for layer in model.layers:
    if layer.name not in [l.name for l in old_model.layers]:
        print("Missing:", layer.name)
for layer in model.layers:
    if layer.name in [l.name for l in old_model.layers]:
        old_layer = old_model.get_layer(layer.name)

        if layer.get_weights():
            new_w = layer.get_weights()
            old_w = old_layer.get_weights()

            if all(n.shape == o.shape for n, o in zip(new_w, old_w)):
                layer.set_weights(old_w)
                print(f"Loaded weights for {layer.name}")
            else:
                print(f"Shape mismatch for {layer.name}")

model.save(modelsPath)
model_json = model.to_json()
with open(os.path.join(modelsPath, "model.json"), "w") as json_file:
    json_file.write(model_json)

print("Model architecture saved.")

# 2️⃣ Save weights
model.save_weights(os.path.join(modelsPath, "model.h5"))

print("Model weights saved.")


full_model = tf.function(lambda x: model(x))

concrete_func = full_model.get_concrete_function(
    tf.TensorSpec(
        shape=(1, 277, 512, 1),   # batch=1
        dtype=tf.float32
    )
)

# Freeze variables into constants
frozen_func = convert_variables_to_constants_v2(concrete_func)

# Get graph definition
graph_def = frozen_func.graph.as_graph_def()

# Save frozen graph
pb_path = os.path.join(modelsPath, "frozen_model13.pb")

tf.io.write_graph(
    graph_def,
    modelsPath,
    "frozen_model13.pb",
    as_text=False
)

print("Frozen graph saved to:")
print(pb_path)

# =====================================================
# PRINT INPUT / OUTPUT NODES
# =====================================================
print("\nInputs:")
print(frozen_func.inputs)

print("\nOutputs:")
print(frozen_func.outputs)