from matplotlib import pyplot as plt
import numpy as np
import random
from modulator import Modulator
from demodulator import Demodulator, PackageState
import pyaudio

import random
import string

letters = string.ascii_lowercase


testN = 1000
testCRCFailed = 0
testPackageFailed = 0
testCorrect = 0

lengDist = [0] * 10


def main():
    global testN
    global testCRCFailed
    global testPackageFailed
    global testCorrect
    global letters
    global lengDist
    # LoRa Parameters

    Fs = 44100
    BW = Fs // 2
    fa = 0
    fb = BW
    SF = 4
    samples = 8192
    Td = samples / Fs

    # Modulator and demodulator
    mod = Modulator(samples, Fs, fa, fb, SF)
    demod = Demodulator(Td, Fs, fa, fb, SF)

    # Preamble and Data
    preamble = [0] * 8

    # printing lowercase
    stringData = "Hola"
    data = mod.stringToSymbols(stringData)
    preamble = np.concatenate((preamble, data))
    y_preamble = mod.getSignal(preamble)

    # Add random start
    # random_totalSamples = random.randint(1, 4000)
    # random_start = np.zeros(random_totalSamples)
    # random_totalSamples = random.randint(1, 4000)
    # random_stop = np.zeros(random_totalSamples)
    # y_preamble = np.concatenate((random_start, y_preamble, random_stop))
    noise_preamble = mod.getNoiseWithSamples(3.0, len(y_preamble))

    timeLength = 1 / len(y_preamble)
    sign = 10 * np.log10(np.sum(y_preamble**2) * timeLength)
    noise = 10 * np.log10(np.sum(noise_preamble**2) * timeLength)
    print(
        "Noise Power: %.3f - Signal Power: %.3f - SNR: %.3f"
        % (noise, sign, sign - noise),
    )
    # y_preamble += noise_preamble

    outSymbols = []

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=Fs,
        output=True,
    )

    stream.write(y_preamble.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    pa.terminate()


# if __name__ == "main":
main()
