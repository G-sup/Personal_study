import tensorflow as tf 

def model_summary():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars,print_info=True)

model_summary()




