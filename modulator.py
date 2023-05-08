"""Module with a LoRa PHY modulator.

It can create a signal given a list of symbols.
"""
import numpy as np
from numpy.typing import NDArray
from crc import Calculator, Crc16


class Modulator:
    """LoRa PHY Modulator"""

    crc_calculator = Calculator(Crc16.CCITT)  # pyright: ignore[reportGeneralTypeIssues]
    g_mat = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def __init__(
        self,
        samples: int,
        sampling_frequency: int,
        init_freq: float,
        end_freq: float,
        spread_factor: int,
        redundancy: int,
    ):
        self.init_freq = init_freq
        self.end_freq = end_freq
        self.spread_factor = spread_factor
        self.sampling_frequency = sampling_frequency
        self.samples = samples
        self.redundancy = redundancy
        self.samples_symbol = samples * redundancy
        self.symbol_step = (end_freq - init_freq) / 2.0**spread_factor
        self.freq_rate = (self.end_freq - self.init_freq) * (self.sampling_frequency / self.samples)
        self.samples_n = 0

    def encode_int(self, value: int) -> int:
        """Encode integer as Hamming Code FEC

        Only implemented for SF=7

        Args:
            value (int): Integer to be encoded

        Returns:
            int: Encoded integer with HammingCode
        """
        output = 0
        bin_value = np.array(np.unpackbits(np.uint8(value))[4:8])
        bin_res = self.g_mat @ bin_value % 2
        for i, val in enumerate(bin_res):
            output += val << 6 - i
        return output

    def str2symbols(self, string: str) -> list[int]:
        """Converts a string to a list of symbols for the given modulator specs

        Args:
            string (str): _description_

        Returns:
            list[int]: _description_
        """

        output: list[int] = []
        temp_symb = self.int2symbols(len(string))
        output.extend(temp_symb)
        for char in string:
            temp_symb = self.char2symbols(char)
            output.extend(temp_symb)
        checksum = self.crc_calculator.checksum(bytes(string, "UTF-8"))
        temp_symb = self.checksum2symbols(checksum)
        output.extend(temp_symb)
        return output

    def int2symbols(self, value: int) -> list[int]:
        """Converts and encode an int value as symbol

        Args:
            value (int): Integer to be converted

        Returns:
            list[int]: List with integers representing the integer as symbols
        """
        output = []
        if self.spread_factor == 8:
            return [value & 0b11111111]
        if self.spread_factor == 7:
            output.append(self.encode_int(value & 0b00001111))
            output.append(self.encode_int((value >> 4) & 0b00001111))
            return output
        if self.spread_factor == 4:
            output.append(value & 0b00001111)
            output.append((value >> 4) & 0b00001111)
            return output
        if self.spread_factor == 2:
            output.append(value & 0b00000011)
            output.append((value >> 2) & 0b00000011)
            output.append((value >> 4) & 0b00000011)
            output.append((value >> 6) & 0b00000011)
            return output
        return []

    def char2symbols(self, char: str) -> list[int]:
        """Converts a single character to list of symbols

        Args:
            char (str): Character to be converted

        Returns:
            list[int]: List with integers representing the integer as symbols
        """
        value = ord(char)
        return self.int2symbols(value)

    def checksum2symbols(self, checksum: int) -> list[int]:
        """Converts the checksum value to symbols

        Args:
            checksum (int): 2-bytes int

        Returns:
            list[int]: List of symbols
        """
        output = []
        first_byte = checksum & 0xFF
        second_byte = (checksum) >> 8 & 0xFF
        output.extend(self.int2symbols(first_byte))
        output.extend(self.int2symbols(second_byte))
        return output

    def symbol2signal(self, symbol: int) -> NDArray[np.float64]:
        """Generates a chirp for the symbol

        Args:
            symbol (int): Symbol

        Raises:
            ValueError: Invalid symbol value

        Returns:
            NDArray[np.float64]: Signal from symbol
        """
        if not symbol < 2.0**self.spread_factor:
            raise ValueError("Symbol must be a valid integer for the selected Spread Factor")
        frac = symbol / 2**self.spread_factor

        first_frac = int(self.samples * (1.0 - frac))
        first_t = np.arange(0, first_frac) / self.sampling_frequency
        freq_offset = (self.end_freq - self.init_freq) * frac + self.init_freq
        first_chirp = np.exp(1j * 2 * np.pi * (first_t * (freq_offset + 0.5 * self.freq_rate * first_t)))

        second_frac = self.samples - first_frac
        second_t = np.arange(0, second_frac) / self.sampling_frequency
        phi0 = first_frac * (freq_offset + 0.5 * self.freq_rate * first_frac)
        second_chirp = np.exp(1j * (2 * np.pi * (second_t * (self.init_freq + 0.5 * self.freq_rate * second_t) + phi0)))

        output: NDArray[np.complex128] = np.concatenate((first_chirp, second_chirp))
        return np.tile(np.real(output), self.redundancy)

    def generate_signal(self, symbols: list) -> NDArray[np.float64]:
        """Generates the whole signal for a list of symbols

        Args:
            symbols (list): List with symbols to convert

        Returns:
            NDArray[np.float64]: Complete signal
        """
        output = []
        codes = len(symbols)
        for symbol in symbols:
            output.extend(self.symbol2signal(symbol))
        self.samples_n = self.samples * codes
        return np.array(output)

    def generate_noise(self, std: float, samples: int | None) -> NDArray[np.float64]:
        """Generates a noisy signal from the std and samples given

        Args:
            std (float): Standard deviation for the noise's normal distribution
            samples (int | None): Number of samples to generate the noise

        Returns:
            NDArray[np.float64]: Noisy signal
        """
        if samples:
            return np.random.normal(0, std, size=samples)
        return np.random.normal(0, std, size=self.samples_n)
