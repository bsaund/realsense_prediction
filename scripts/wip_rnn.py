import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from realsense_prediction import datatools, model_runner, utils
from arc_utilities import video_writer
from realsense_prediction.simple_convlstm import get_simple_convlstm, generate_movies, SimpleConvLstm, MyModel


# keras.layers.TimeDistributed


def main():
    myModel = MyModel()
    myModel.compile(loss="binary_crossentropy", optimizer="adadelta")
    model = get_simple_convlstm()
    m2 = SimpleConvLstm(hparams={}, batch_size=10)
    m2.compile(loss="binary_crossentropy", optimizer="adadelta")
    path = datatools.get_trial_path() / 'trial_01'
    mr = model_runner.ModelRunner(model, True, path, None)


    print('generating movies')
    noisy_movies, shifted_movies = generate_movies()
    a_movie = noisy_movies[0]
    some_movies = noisy_movies[0:2]

    # model.fit(noisy_movies, shifted_movies, batch_size=10,
    #           epochs=1,
    #           verbose=2,
    #           validation_split=0.1)
    #
    m2.fit(noisy_movies, shifted_movies, batch_size=10, epochs=1, verbose=2, validation_split=0.1)
    output = model(noisy_movies[0])




if __name__ == '__main__':
    main()
