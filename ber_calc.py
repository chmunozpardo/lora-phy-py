import matplotlib.pyplot as plt
import numpy as np
import random
from modulator import Modulator
from demodulator import Demodulator
from scipy.signal import stft

import random

testN = 1000
berResult = 0.0


def main():
    global berResult

    # LoRa Parameters
    BW = 22000
    Fs = 2 * BW
    fa = 0
    fb = BW
    SF = 4
    samples = 8192
    Td = samples / Fs

    # Modulator and demodulator
    mod = Modulator(samples, Fs, fa, fb, SF)
    demod = Demodulator(samples, Fs, fa, fb, SF)

    # Preamble and Data
    value = random.randint(0, 2**SF - 1)
    y_preamble = mod.getSignal([value])
    noise_preamble = mod.getNoiseWithSamples(6.0, len(y_preamble))
    timeLength = 1 / len(y_preamble)
    sign = 10 * np.log10(np.sum(y_preamble**2) * timeLength)
    noise = 10 * np.log10(np.sum(noise_preamble**2) * timeLength)
    print(
        "Noise Power: %.3f - Signal Power: %.3f - SNR: %.3f"
        % (noise, sign, sign - noise),
    )
    y_preamble += noise_preamble

    leng = int(np.floor(len(y_preamble) / samples))
    for index in range(leng):
        result = demod.detectSymbol(
            y_preamble[index * samples : samples * (index + 1)]
        )
        if result:
            r = (1 << np.arange(8))[:, None]
            ber = np.count_nonzero((result & r) != (value & r)) / SF
            berResult += ber


for i in range(testN):
    print("%3d" % (i), end=" - ")
    main()

print(
    "Ber result: %.4f%%" % (berResult / testN * 100.0),
)
