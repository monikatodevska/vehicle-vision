model_path=r'C:\Users\User\Desktop\desktop\keras-tf-pb-master\models_tmp\koli\modelAnchorless_koli.pb'
# model_path=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor_Renamed_PRVVTORsloj_16filters_BEZ_SKIP_1INC_V3\model.json'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow.keras.models import model_from_config


# import tensorflow as tf

import tensorflow as tf

# Load the TensorFlow graph from the .pb file
graph = tf.Graph()
with graph.as_default():
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as file:
        graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name='')

# Remove the placeholder 'input_2' from the graph
graph_def = tf.compat.v1.graph_util.remove_training_nodes(graph_def, ['input_2'])

# Print all operation names in the modified graph
with tf.compat.v1.Session(graph=graph) as sess:
    tf.import_graph_def(graph_def, name='')
    ops = [op.name for op in graph.get_operations()]
    print(ops)

# Get the input and output nodes
input_node_name = 'input_2:0'
output_class_node_name = 'out_class/BiasAdd:0'
output_reg_node_name = 'out_reg/BiasAdd:0'

# Create a session to run the TensorFlow graph
with tf.compat.v1.Session(graph=graph) as sess:
    # Get the input and output tensors
    input_tensor = graph.get_tensor_by_name(input_node_name)
    output_class_tensor = graph.get_tensor_by_name(output_class_node_name)
    output_reg_tensor = graph.get_tensor_by_name(output_reg_node_name)

    # Create a Keras model
    keras_model = tf.keras.models.Model(inputs=input_tensor, outputs=[output_class_tensor, output_reg_tensor])

# Save the Keras model to a file
keras_model.save('path/to/keras_model.h5')



#
#
# import tensorflow as tf
# from tensorflow import saved_model
# from tensorflow.keras.utils import plot_model
#
# # Load the SavedModel
# # model = tf.keras.models.load_model(model_path)
# model = tf.keras.models.load_model(model_path)
# # model = saved_model.load(model_path,None,r'C:\Users\User\Desktop\desktop\keras-tf-pb-master\models_tmp\koli')
#
# # Plot the model architecture
# plot_model(model, to_file='model_architecture.png', show_shapes=True)