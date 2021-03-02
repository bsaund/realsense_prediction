"""
Title: Next-frame prediction with Conv-LSTM
Author: [jeammimi](https://github.com/jeammimi)
Date created: 2016/11/02
Last modified: 2020/05/01
Description: Predict the next frame in a sequence using a Conv-LSTM model.
"""
"""
## Introduction
This script demonstrates the use of a convolutional LSTM model.
The model is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""

"""
## Setup
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pylab as plt
from arc_utilities import video_writer
from pathlib import Path
from realsense_prediction.simple_convlstm import get_simple_convlstm, generate_movies

"""
## Build a model
We create a model which take as input movies of shape
`(n_frames, width, height, channels)` and returns a movie
of identical shape.
"""


seq = get_simple_convlstm()

"""
## Generate artificial data
Generate movies with 3 to 7 moving squares inside.
The squares are of shape 1x1 or 2x2 pixels,
and move linearly over time.
For convenience, we first create movies with bigger width and height (80x80)
and at the end we select a 40x40 window.
"""





"""
## Train the model
"""

epochs = 100  # In practice, you would need hundreds of epochs.
print('generating movies')
noisy_movies, shifted_movies = generate_movies(n_samples=1200)

print('fitting model')
seq.fit(
    noisy_movies[:1000],
    shifted_movies[:1000],
    batch_size=10,
    epochs=epochs,
    verbose=2,
    validation_split=0.1,
)
print('testing model')
"""
## Test the model on one movie
Feed it with the first 7 positions and then
predict the new positions.
"""

movie_index = 1004
test_movie = noisy_movies[movie_index]

# Start from first 7 frames
track = test_movie[:7, ::, ::, ::]
video_writer.save_video(test_movie * 255,
                        filepath=Path.home().as_posix() + '/tmp/test_movie.mp4')

# Predict 16 frames
for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
video_writer.save_video(track * 255,
                        filepath=Path.home().as_posix() + '/tmp/track.mp4')
