import numpy as np
from crc import CrcCalculator, Crc16


class Modulator:
    crc_calculator = CrcCalculator(Crc16.CCITT)

    def __init__(
        self, samples: float, Fs: int, fa: float, fb: float, SF: int
    ) -> None:
        self.fa = fa
        self.fb = fb
        self.SF = SF
        self.Fs = Fs
        self.samples = samples
        self.symbolStep = (fb - fa) / 2.0**SF
        self.totalSamples = 0
        pass

    def stringToSymbols(self, string: str) -> list:
        symbols = []
        tempSymbols = self.lengthToSymbols(len(string))
        for symb in tempSymbols:
            symbols.append(symb)
        for char in string:
            tempSymbols = self.charToSymbols(char)
            for symb in tempSymbols:
                symbols.append(symb)
        checksum = self.crc_calculator.calculate_checksum(
            bytes(string, "UTF-8")
        )
        tempSymbols = self.checksumToSymbols(checksum)
        for symb in tempSymbols:
            symbols.append(symb)
        return symbols

    def lengthToSymbols(self, length: int):
        lowNible = length & 0x0F
        highNible = (length >> 4) & 0x0F
        return [lowNible, highNible]

    def charToSymbols(self, char: str):
        value = ord(char)
        lowNible = value & 0x0F
        highNible = (value >> 4) & 0x0F
        return [lowNible, highNible]

    def checksumToSymbols(self, checksum: int):
        firstNible = checksum & 0x0F
        secondNible = (checksum >> 4) & 0x0F
        thirdNible = (checksum >> 8) & 0x0F
        fourthNible = (checksum >> 12) & 0x0F
        return [firstNible, secondNible, thirdNible, fourthNible]

    def getSymbolSignal(self, symbol: int):
        assert symbol < 2.0**self.SF
        frac = symbol / 2**self.SF
        th = int(self.samples * (1.0 - frac))
        sh = self.samples - th
        t0 = np.arange(0, th) / self.Fs
        t1 = np.arange(0, sh) / self.Fs

        freqRate = (self.fb - self.fa) * (self.Fs / self.samples)
        fh = (self.fb - self.fa) * frac

        c0 = np.exp(
            1j * 2 * np.pi * (t0 * (self.fa + fh + 0.5 * freqRate * t0))
        )
        phi0 = 2 * np.pi * (th * (self.fa + fh + 0.5 * freqRate * th))
        c1 = np.exp(
            1j * (2 * np.pi * (t1 * (self.fa + 0.5 * freqRate * t1)) + phi0)
        )

        cf = np.concatenate((c0, c1))
        return cf.real

    def getSignal(self, symbols: list):
        y = np.array([])
        codes = len(symbols)
        for symbol in symbols:
            y = np.concatenate((y, self.getSymbolSignal(symbol)))
        self.totalSamples = self.samples * codes
        return y

    def getNoise(self, std: float):
        return np.random.normal(0, std, size=self.totalSamples)

    def getNoiseWithSamples(self, std: float, samples: int):
        return np.random.normal(0, std, size=samples)
