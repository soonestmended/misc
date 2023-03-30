import tensorflow as tf
print("Num GPUs available: ", len(tf.config.list_physical_devices('GPU')))

