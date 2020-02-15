import pydub
import numpy as np
from scipy.io import wavfile
from audio2numpy import open_audio


def read(filename, normalized=False):
    signal, sampling_rate = open_audio(filename)

    return signal, sampling_rate
    # fs, data = wavfile.read(filename)
    # return fs, data
    # """MP3 to numpy array"""
    # a = pydub.AudioSegment.from_mp3(f)
    # y = np.array(a.get_array_of_samples())
    # if a.channels == 2:
    #     y = y.reshape((-1, 2))
    # if normalized:
    #     return a.frame_rate, np.float32(y) / 2 ** 15
    # else:
    #     return a.frame_rate, y


def main():
    filename = "./audio/spanish/acceso.mp3"
    a, b = read(filename)


if __name__ == "__main__":
    main()
