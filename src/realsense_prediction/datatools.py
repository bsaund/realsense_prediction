import tensorflow as tf
import rospy
import rospkg
import yaml
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def load_config():
    config_fp = get_base_path() / 'config.yaml'
    with config_fp.open() as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def get_base_path():
    return Path(rospkg.RosPack().get_path('realsense_prediction'))


def get_trial_path():
    return get_base_path() / "trials"


def load_movie():
    data_path = Path(load_config()['dataset_path'])
    first_movie = sorted([p for p in data_path.iterdir()])[0]
    frames = sorted([f for f in first_movie.iterdir() if f.suffix == '.jpg'])
    arrs = []
    for i, frame in enumerate(frames[1:100]):
        # print(f'loading frame {i}')
        img = img_to_array(load_img(frame))
        arrs.append(tf.image.resize(img, [48, 64]))
    # return np.array([arrs])
    return tf.stack([tf.stack(arrs)])

    # return data_path.glob()
