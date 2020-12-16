import tensorflow as tf
from batch_rl.baselines.train import main
import dopamine


def print_gpus():
    ld = tf.config.list_logical_devices()
    for ldi in ld:
        print(ldi)


if __name__ == '__main__':
    # print_gpus()
    # print(dopamine.name)
    main()
