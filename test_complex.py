import numpy as np
import random
from modulator import Modulator
from demodulator import Demodulator, PackageState

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

    BW = 22000
    Fs = 2 * BW
    fa = 0
    fb = BW
    SF = 4
    samples = 8192
    Td = samples / Fs

    # Modulator and demodulator
    mod = Modulator(Td, Fs, fa, fb, SF)
    demod = Demodulator(Td, Fs, fa, fb, SF)

    # Preamble and Data
    preamble = [0] * 12
    preamble[3] = 1

    # printing lowercase

    dataLength = random.randint(1, 100)
    stringData = "".join(random.choice(letters) for i in range(dataLength))
    data = mod.stringToSymbols(stringData)
    preamble = np.concatenate((preamble, data))
    y_preamble = mod.getSignal(preamble)

    # Add random start
    random_totalSamples = random.randint(1, 4000)
    random_start = np.zeros(random_totalSamples)
    random_totalSamples = random.randint(1, 4000)
    random_stop = np.zeros(random_totalSamples)
    y_preamble = np.concatenate((random_start, y_preamble, random_stop))
    noise_preamble = mod.getNoiseWithSamples(3.0, len(y_preamble))

    timeLength = 1 / len(y_preamble)
    sign = 10 * np.log10(np.sum(y_preamble**2) * timeLength)
    noise = 10 * np.log10(np.sum(noise_preamble**2) * timeLength)
    print(
        "Noise Power: %.3f - Signal Power: %.3f - SNR: %.3f"
        % (noise, sign, sign - noise),
    )
    y_preamble += noise_preamble

    outSymbols = []

    leng = int(np.floor(len(y_preamble) / samples))
    for index in range(leng):
        result = demod.detectSymbol(
            y_preamble[index * samples : samples * (index + 1)]
        )
        demod.stateMachine(result)
        outSymbols.append(result)

    if demod.package["status"] == PackageState.PACKAGE_DETECTED:
        testPackageFailed += 1
        lengDist[(dataLength - 1) // 10] += 1
    if demod.package["status"] == PackageState.NO_PACKAGE:
        testPackageFailed += 1
        lengDist[(dataLength - 1) // 10] += 1
    elif demod.package["status"] == PackageState.CRC_ERROR:
        testCRCFailed += 1
        lengDist[(dataLength - 1) // 10] += 1
    elif demod.package["status"] == PackageState.PACKAGE_READY:
        testCorrect += 1
    print(demod.package["status"], dataLength)


for i in range(testN):
    print("%3d" % (i), end=" - ")
    main()

print(
    "CRC fail: %.3f%%\nPackage fail: %.3f%%\nCorrect: %.3f%%"
    % (
        testCRCFailed / testN * 100.0,
        testPackageFailed / testN * 100.0,
        testCorrect / testN * 100.0,
    ),
)

print(lengDist)
