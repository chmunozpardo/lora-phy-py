import numpy as np
from enum import Enum
from crc import CrcCalculator, Crc16


class PackageState(Enum):
    NO_PACKAGE = 0
    CRC_ERROR = 1
    UNCOMPLETED_LENGTH = 2
    PACKAGE_DETECTED = 3
    PACKAGE_READY = 4


class StateMachine(Enum):
    IDLE = 0
    PREAMB1 = 1
    PREAMB2 = 2
    PREAMB3 = 3
    PREAMB4 = 4
    PREAMB5 = 5
    PREAMB6 = 6
    PREAMB7 = 7
    PREAMB8 = 8
    LENG1 = 9
    LENG2 = 10
    DATA = 11
    CHECKSUM1 = 12
    CHECKSUM2 = 13
    CHECKSUM3 = 14
    CHECKSUM4 = 15
    SKIP_NEXT = 16


class Demodulator:
    crc_calculator = CrcCalculator(Crc16.CCITT)

    def __init__(
        self, samples: int, Fs: int, fa: float, fb: float, SF: int
    ) -> None:
        self.fa = fa
        self.fb = fb
        self.SF = SF
        self.Fs = Fs
        self.df = fb - fa
        self.samples = samples
        self.symbolStep = (fb - fa) / 2.0**SF
        self.totalSamples = 0
        self.window = np.hanning(self.samples)
        self.binSize = int(self.samples / 2 ** (SF + 1))
        t0 = np.arange(0, self.samples) / self.Fs

        freqRate = (self.fb - self.fa) * (self.Fs / self.samples)
        self.upchirp = np.exp(
            1j * 2 * np.pi * (t0 * (self.fa + 0.5 * freqRate * t0))
        )
        self.downchirp = np.exp(
            -1j * 2 * np.pi * (t0 * (self.fa + 0.5 * freqRate * t0))
        )

        self.currentState = StateMachine.IDLE
        self.lastSymbol = None

        self.debugList = []
        self.dataCounter = 0
        self.resultLength = 0
        self.resultSymbols = []
        self.checksum = []
        self.package = {
            "status": PackageState.NO_PACKAGE,
            "value": "",
            "checksum": 0,
        }
        pass

    def symbolsToString(self, symbols: list) -> list:
        self.string = ""
        for index in range(0, len(symbols), 2):
            self.string += self.symbolsToChar(symbols[index : index + 2])
        return self.string

    def symbolsToChar(self, symbols: list):
        value = symbols[0]
        value += symbols[1] << 4
        return chr(value)

    def detectSymbol(self, signal: list):
        assert len(signal) == self.samples
        midSample = self.samples // 2
        result = signal * self.downchirp * self.window
        fftTemp = np.zeros(midSample)
        fftResult = np.fft.fft(result)
        fftAbs = np.abs(fftResult) / self.samples
        for binIndex in range(2**self.SF):
            startInd = binIndex * self.binSize
            endInd = startInd + self.binSize
            fftTemp[startInd:endInd] = (
                fftAbs[startInd:endInd]
                + fftAbs[midSample + startInd : (midSample + endInd)]
            )
        peak = np.argmax(fftTemp)
        if fftTemp[peak] < 0.01:
            return None
        return peak // self.binSize

    def stateMachine(self, symbol: int):
        if self.currentState == StateMachine.IDLE:
            if symbol is not None:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB1
                return
            else:
                return
        elif self.currentState == StateMachine.PREAMB1:
            if symbol is not None and self.lastSymbol == symbol:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB2
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.PREAMB2:
            if symbol is not None and self.lastSymbol == symbol:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB3
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.PREAMB3:
            if symbol is not None and self.lastSymbol == symbol:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB4
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.PREAMB4:
            if symbol is not None and self.lastSymbol == symbol:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB5
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.PREAMB5:
            if symbol is not None and self.lastSymbol == symbol:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB6
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.PREAMB6:
            if symbol is not None and self.lastSymbol == symbol:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB7
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.PREAMB7:
            if symbol is not None and self.lastSymbol == symbol:
                self.lastSymbol = symbol
                self.currentState = StateMachine.PREAMB8
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.PREAMB8:
            if symbol is not None:
                self.resultLength = (symbol - self.lastSymbol) % 2**self.SF
                self.currentState = StateMachine.LENG1
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.LENG1:
            if symbol is not None:
                self.resultLength += (
                    (symbol - self.lastSymbol) % 2**self.SF
                ) << 4
                if self.resultLength == 0:
                    self.lastSymbol = None
                    self.currentState = StateMachine.IDLE
                    return
                self.resultLength *= 2
                self.currentState = StateMachine.LENG2
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.LENG2:
            if symbol is not None:
                self.resultSymbols = []
                self.resultSymbols.append(
                    (symbol - self.lastSymbol) % 2**self.SF
                )
                self.currentState = StateMachine.DATA
                self.dataCounter += 1
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.DATA:
            if symbol is not None:
                self.resultSymbols.append(
                    (symbol - self.lastSymbol) % 2**self.SF
                )
                self.currentState = StateMachine.DATA
                self.dataCounter += 1
                if self.dataCounter == self.resultLength:
                    value = self.symbolsToString(self.resultSymbols)
                    self.package = {
                        "status": PackageState.PACKAGE_DETECTED,
                        "value": value,
                        "checksum": self.crc_calculator.calculate_checksum(
                            bytes(value, "UTF-8")
                        ),
                    }
                    self.currentState = StateMachine.CHECKSUM1
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.CHECKSUM1:
            if symbol is not None:
                self.checksum = (symbol - self.lastSymbol) % 2**self.SF
                self.currentState = StateMachine.CHECKSUM2
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.CHECKSUM2:
            if symbol is not None:
                self.checksum += (
                    (symbol - self.lastSymbol) % 2**self.SF
                ) << 4

                self.currentState = StateMachine.CHECKSUM3
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.CHECKSUM3:
            if symbol is not None:
                self.checksum += (
                    (symbol - self.lastSymbol) % 2**self.SF
                ) << 8
                self.currentState = StateMachine.CHECKSUM4
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.CHECKSUM4:
            if symbol is not None:
                self.checksum += (
                    (symbol - self.lastSymbol) % 2**self.SF
                ) << 12
                value = self.package["value"]
                packageChecksum = self.package["checksum"]
                if self.checksum == self.package["checksum"]:
                    self.package = {
                        "status": PackageState.PACKAGE_READY,
                        "value": value,
                        "checksum": packageChecksum,
                    }
                else:
                    self.package = {
                        "status": PackageState.CRC_ERROR,
                        "value": value,
                        "checksum": [self.checksum, packageChecksum],
                    }
                self.currentState = StateMachine.SKIP_NEXT
                return
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
            return
        elif self.currentState == StateMachine.SKIP_NEXT:
            self.lastSymbol = None
            self.dataCounter = 0
            self.resultLength = 0
            self.currentState = StateMachine.IDLE
