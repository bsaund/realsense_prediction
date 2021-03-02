import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from realsense_prediction import datatools
from arc_utilities import video_writer

# keras.layers.TimeDistributed

seq = keras.Sequential(
    [
        keras.Input(
            shape=(None, 48, 64, 3)
        ),  # Variable-length sequence of 40x40x1 frames
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=3, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        ),
    ]
)
seq.compile(loss="binary_crossentropy", optimizer="adadelta")

if __name__ == '__main__':
    data = datatools.load_movie()
    video_writer.save_video(data[0, :, :, :, :], '/home/bsaund/tmp/rnn.mp4', fps=10)
    print(data.shape)
    print("Training")
    # seq.fit(data, data, batch_size=1, epochs=100, verbose=2)
