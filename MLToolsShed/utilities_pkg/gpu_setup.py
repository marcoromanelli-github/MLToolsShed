"""
Setup GPU before usage: this must be the first command before starting to set up networks otherwise it does not know where
to allocate memory
"""
import os
import tensorflow as tf


def gpu_setup_v2(id_gpu, memory_limit_bytes=None):
    """
    :param id_gpu: integer id of the selected gpu
    :param memory_limit_bytes: upper bound for the bytes of memory available
    :return: None
    """
    gpus = tf.test.gpu_device_name()
    print(gpus)
    print("Num GPUs Available: ", len(gpus))
    if isinstance(id_gpu, int) and id_gpu < len(gpus):
        # Restrict TensorFlow to only use the GPU with id id_gpu
        try:
            tf.compat.v1.config.experimental.set_visible_devices(gpus[id_gpu], 'GPU')
            tf.compat.v1.config.experimental.set_memory_growth(gpus[id_gpu], True)
            if memory_limit_bytes is not None:
                tf.compat.v1.config.experimental.set_virtual_device_configuration(gpus[id_gpu], [
                    tf.compat.v1.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_bytes)])
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    return None


def gpu_setup_v1(id_gpu, memory_percentage):
    """
    :param id_gpu: integer id of the selected gpu
    :param memory_percentage: float in [0,1]
    :return: None
    """
    gpus = tf.test.gpu_device_name()
    print(gpus)
    print("Num GPUs Available: ", len(gpus))

    if isinstance(id_gpu, int) and id_gpu < len(gpus):
        id_gpu = str(id_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = id_gpu
        #   TensorFlow wizardry
        configuration = tf.compat.v1.ConfigProto()

        #   Don't pre_allocate memory; allocate as_needed
        configuration.gpu_options.allow_growth = True

        #   Only allow a total of half the GPU memory to be allocated
        #   memory_percentage is a float between 0 and 1
        configuration.gpu_options.per_process_gpu_memory_fraction = memory_percentage

        #   Create a session with the options specified above
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=configuration))

    return None
