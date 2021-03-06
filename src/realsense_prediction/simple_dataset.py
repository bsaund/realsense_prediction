from itertools import zip_longest
import numpy as np




def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



class SimpleDataset:
    def __init__(self, metadataset=None):
        self.metadataset = metadataset

    def set_num_samples(self, num_samples):
        self.metadataset = [i for i in range(num_samples)]
        return self

    def batch(self, batch_size, skip_uneven=True):
        return [SimpleDataset(seq) for seq in chunker(self.metadataset, batch_size)
                if len(seq) == batch_size or not skip_uneven]

    def load(self):
        noisy_movies, shifted_movies = generate_movies(len(self.metadataset))
        return {'input': noisy_movies, 'output': shifted_movies}


def generate_movies(n_samples=1200, n_frames=30):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float32)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float32)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[
                i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0
                ] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the model to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1) ** np.random.randint(0, 2)
                    noisy_movies[
                    i,
                    t,
                    x_shift - w - 1: x_shift + w + 1,
                    y_shift - w - 1: y_shift + w + 1,
                    0,
                    ] += (noise_f * 0.1)

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[
                i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0
                ] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies