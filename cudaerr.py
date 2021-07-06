# 뭔가 GPU(CUDA) 문제인듯 한거에서 해결
# https://www.tensorflow.org/guide/gpu
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])