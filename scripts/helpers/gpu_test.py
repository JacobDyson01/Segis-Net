import tensorflow as tf

# TensorFlow Section
def check_tensorflow():
    print("Checking TensorFlow:")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # List available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"TensorFlow is using GPU: {physical_devices[0].name}")
        print(f"CUDA Version: {tf.sysconfig.get_build_info()['cuda_version']}")
        print(f"CUDNN Version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    else:
        print("CUDA is NOT available for TensorFlow.")

if __name__ == '__main__':
    check_tensorflow()
