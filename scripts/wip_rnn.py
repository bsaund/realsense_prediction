import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from realsense_prediction import datatools, model_runner, utils
from arc_utilities import video_writer
from realsense_prediction.simple_convlstm import get_simple_convlstm, SimpleConvLstm, MyModel
from realsense_prediction.simple_dataset import SimpleDataset
from realsense_prediction.utils import filepath_utils
import argparse
from pathlib import Path


def train(group):
    # myModel = MyModel()
    ds = SimpleDataset()
    ds.set_num_samples(10000)
    ds = ds.batch(10)

    val_ds = SimpleDataset()
    val_ds.set_num_samples(100)
    val_ds = val_ds.batch(10)

    model = SimpleConvLstm(hparams={}, batch_size=None)
    path, _ = filepath_utils.create_trial(group, params={})
    mr = model_runner.ModelRunner(model, True, path, params=None)
    mr.train(ds, val_ds, num_epochs=100)


def test(group, trial):
    print(f'loading group {group}, trial {trial}')
    model_path, params = filepath_utils.load_trial(Path(group) / trial)
    model = SimpleConvLstm(hparams=params, batch_size=None)
    untrained_model = SimpleConvLstm(hparams=params, batch_size=None)
    mr = model_runner.ModelRunner(model, False, model_path, params,
                                  checkpoint=model_path / 'latest_checkpoint')
    test_ds = SimpleDataset().set_num_samples(100)
    test_ds = test_ds.batch(1)
    test_elem = test_ds[0].load()
    mr.model(test_elem)

    tf.reduce_max(untrained_model(test_elem) - test_elem['output'])
    tf.reduce_max(model(test_elem) - test_elem['output'])

    for i in range(len(model.weights)):
        print(f'layer {i} max weights are {tf.reduce_max(model.weights[i])}, '
              f'{tf.reduce_max(untrained_model.weights[i])}')

    forward_video = forward_predict_movie(mr.model, test_elem)
    video_writer.save_video(forward_video['input'][0], Path().home() / 'tmp' / 'gen.mp4')


def forward_predict_movie(model, track, run_in_steps=10, num_steps=40):
    print("Forward predicting movie")
    track['input'] = track['input'][:, 0:run_in_steps, :, :, :]
    for j in range(num_steps):
        new_pos = model(track)
        new = new_pos[::, -1:, ::, ::, ::]
        track['input'] = tf.concat((track['input'], new), axis=1)
    return track


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--test', action='store_true')
    p.add_argument('--group', default='wip')
    p.add_argument('--trial')
    args = p.parse_args()

    if args.train:
        train(args.group)
    if args.test:
        test(args.group, args.trial)
    print(args.group)
