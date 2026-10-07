import tensorflow as tf
from keras.models import model_from_json
import os
# model_path=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva_yolo_jovan'
# model_path=r'D:\Tatjana\NewBeginning\MachineVision\Models\predniciM_v1'
# model_path=r'D:\Science\AICityChallenge2023\Models\contrastiveloss_generator_8nka_plus_4300_with_flipped_NOAVATARS_norm_v2_INIT2'
model_path=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva\new_189'


def crop_to_match(inputs):
    skip, f9 = inputs
    skip_height = tf.shape(skip)[1]
    f9_height = tf.shape(f9)[1]
    crop_top = skip_height - f9_height
    return skip[:, crop_top:, :, :]

# Step 1: Load the JSON model architecture
with open(os.path.join(model_path,'model.json'), 'r') as json_file:
    model_json = json_file.read()

# Recreate the model architecture from the JSON file
model = model_from_json(
    model_json,
    custom_objects={"crop_to_match": crop_to_match}
)

# Step 2: Load the model weights
model.load_weights(os.path.join(model_path,'model.h5'))
#
# # Step 3: Save the model in the SavedModel format
saved_model_dir = os.path.join(model_path)

model.save(saved_model_dir, save_format="tf")
# Step 4: The saved_model.pb file is created in the saved_model/ directory
# import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Convert the model to a concrete function
full_model = tf.function(lambda x: model(x))
concrete_func = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Convert variables to constants (freeze the model)
frozen_func = convert_variables_to_constants_v2(concrete_func)
graph_def = frozen_func.graph.as_graph_def()

# Save the frozen graph
tf.io.write_graph(graph_def, '.', os.path.join(model_path, 'frozen_model7.pb'), as_text=False)
