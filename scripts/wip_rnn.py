import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from realsense_prediction import datatools, model_runner, utils
# from arc_utilities import video_writer
from realsense_prediction.simple_convlstm import get_simple_convlstm, SimpleConvLstm, MyModel
from realsense_prediction.simple_dataset import SimpleDataset
from realsense_prediction.utils import filepath_utils

# keras.layers.TimeDistributed


def main():
    # myModel = MyModel()
    ds = SimpleDataset()
    ds.set_num_samples(1000)
    ds = ds.batch(10)

    model = SimpleConvLstm(hparams={}, batch_size=None)
    # myModel.compile(loss="binary_crossentropy", optimizer="adadelta")
    # path = datatools.get_trial_path() / 'wip'/ 'trial_01'
    path, _ = filepath_utils.create_trial('wip', params = {})
    mr = model_runner.ModelRunner(model, True, path, params=None)
    mr.train(ds, ds, num_epochs=10)

    # myModel.fit(np.random.random((5, 10,40,40,1)), np.random.random((5, 10,40,40,1)))
    # model = get_simple_convlstm()
    # m2 = SimpleConvLstm(hparams={}, batch_size=10)
    # m2.compile(loss="binary_crossentropy", optimizer="adadelta")
    # path = datatools.get_trial_path() / 'trial_01'
    # mr = model_runner.ModelRunner(model, True, path, None)

    print('generating movies')
    # noisy_movies, shifted_movies = generate_movies()
    # myModel.fit(noisy_movies, shifted_movies, batch_size=5)
    # a_movie = noisy_movies[0]
    # some_movies = noisy_movies[0:2]

    # model.fit(noisy_movies, shifted_movies, batch_size=10,
    #           epochs=1,
    #           verbose=2,
    #           validation_split=0.1)
    #
    # m2.fit(noisy_movies, shifted_movies, batch_size=10, epochs=1, verbose=2, validation_split=0.1)
    # output = model(noisy_movies[0])


if __name__ == '__main__':
    main()
