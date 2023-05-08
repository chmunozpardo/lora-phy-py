"""Module with a LoRa PHY demodulator.

Takes a signal and it converts it to
"""
from enum import Enum
from dataclasses import dataclass
import numpy as np
from crc import Calculator, Crc16


class PackageState(Enum):
    """Package State Enum"""

    NO_PACKAGE = 0
    CRC_ERROR = 1
    UNCOMPLETED_LENGTH = 2
    PACKAGE_DETECTED = 3
    PACKAGE_READY = 4


class StateMachine(Enum):
    """Demodulator State Machine Enum"""

    IDLE = 1
    PREAMB = 2
    LENG = 3
    DATA = 4
    CHECKSUM = 5
    SKIP_NEXT = 6


@dataclass
class PackageInfo:
    """Contains information of the last package received"""

    status: PackageState = PackageState.NO_PACKAGE
    value: str = ""
    checksum: int = 0


class Demodulator:
    """LoRa PHY Demodulator"""

    crc_calculator = Calculator(Crc16.CCITT)  # pyright: ignore[reportGeneralTypeIssues]
    package: PackageInfo
    last_package: PackageInfo
    h_mat = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
    p_mat = np.array([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

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
        self.total_samples = 0
        self.window = np.hanning(self.samples_symbol)
        self.bin_size = int(self.samples / 2 ** (spread_factor + 1))
        t_array = np.arange(0, self.samples) / self.sampling_frequency
        freq_rate = (self.end_freq - self.init_freq) * (self.sampling_frequency / self.samples)
        upchirp = np.exp(1j * 2 * np.pi * (t_array * (self.init_freq + 0.5 * freq_rate * t_array)))
        self.upchirp = np.tile(upchirp, self.redundancy)
        downchirp = np.exp(-1j * 2 * np.pi * (t_array * (self.init_freq + 0.5 * freq_rate * t_array)))
        self.downchirp = np.tile(downchirp, self.redundancy)

        self.current_state = StateMachine.IDLE
        self.reference_symbol = 0
        self.last_symbol = None

        self.preamb_counter = 0
        self.data_counter = 0
        self.length_counter = 0
        self.checksum_counter = 0

        self.result_length = 0
        self.result_symbols = []
        self.checksum = []
        self.package = PackageInfo()
        self.preamb_size = 8

        if self.spread_factor == 2:
            self.step = 4
            self.leng_size = 4
            self.checksum_size = 8
        elif self.spread_factor == 4:
            self.step = 2
            self.leng_size = 2
            self.checksum_size = 4
        elif self.spread_factor == 8:
            self.step = 1
            self.leng_size = 1
            self.checksum_size = 2
        else:
            self.step = 0

    def decode_int(self, value: int) -> int:
        """Decode integer as Hamming Code FEC

        Only implemented for SF=7

        Args:
            value (int): Integer to be decoded

        Returns:
            int: Decoded integer from HammingCode
        """
        output = 0
        if self.spread_factor == 7:
            bin_value = np.array(np.unpackbits(np.uint8(value))[1:8])
            bin_res = self.h_mat @ bin_value % 2
            for i, val in enumerate(bin_res):
                output += val << 2 - i
            if output > 0:
                bin_value[output - 1] ^= 0b1
            bin_res = self.p_mat @ bin_value
            output = 0
            for i, val in enumerate(bin_res):
                output += val << 3 - i
            return output
        return value

    def symbols2string(self, symbols: list) -> str:
        """Converts a list of symbols to a string

        Args:
            symbols (list): Symbols to be converted

        Returns:
            str: Decoded string
        """
        output = ""

        for index in range(0, len(symbols), self.step):
            output += self.symbols2char(symbols[index : index + self.step])
        return output

    def symbols2char(self, symbols: list[int]) -> str:
        """Converts a list of symbols to a char

        Args:
            symbols (list): Symbols to be converted

        Returns:
            str: Single char
        """
        if self.spread_factor == 8:
            value = symbols[0]
        elif self.spread_factor == 7:
            value = self.decode_int(symbols[0])
            value += self.decode_int(symbols[1] << 4)
        elif self.spread_factor == 4:
            value = symbols[0]
            value += symbols[1] << 4
        elif self.spread_factor == 2:
            value = symbols[0]
            value += symbols[1] << 2
            value += symbols[2] << 4
            value += symbols[3] << 6
        else:
            return ""
        return chr(value)

    def detect_symbol(self, signal: list[float]) -> int | None:
        """Checks if in the input signal there is a valid symbol

        Args:
            signal (list[float]): Input signal

        Raises:
            ValueError: Length of signal must be equal to the samples of the demodulator

        Returns:
            int | None: Returns the symbol number if it can be detected. Otherwise returns None
        """
        if not len(signal) == self.samples * self.redundancy:
            raise ValueError
        mid_sample = self.samples // 2
        result = signal * self.downchirp * self.window
        fft_temp = np.zeros(mid_sample)
        for redundancy_ind in range(self.redundancy):
            fft_partial = result[self.samples * redundancy_ind : self.samples * (redundancy_ind + 1)]
            fft_result = np.fft.fft(fft_partial) / self.samples
            fft_abs = np.abs(fft_result)
            for bin_ind in range(2**self.spread_factor):
                start_ind = self.bin_size * bin_ind
                end_ind = self.bin_size * (bin_ind + 1)
                fft_temp[start_ind:end_ind] += (
                    fft_abs[start_ind:end_ind] + fft_abs[(mid_sample + start_ind) : (mid_sample + end_ind)]
                )
        comp_temp = fft_temp
        peak = np.argmax(comp_temp)
        avg_fft = np.mean(comp_temp)
        std_fft = np.std(comp_temp)
        if comp_temp[peak] < (avg_fft + 2.0 * std_fft):
            return None
        return int(peak) // self.bin_size

    def reset_machine(self, symbol: int | None = None):
        """Resets the demodulator state machine"""
        self.last_symbol = symbol
        self.preamb_counter = 0
        self.data_counter = 0
        self.length_counter = 0
        self.checksum_counter = 0
        self.result_length = 0
        self.current_state = StateMachine.IDLE
        self.package = PackageInfo()

    def state_machine(self, symbol: int | None):
        """Goes to the next state depending on symbol"""
        if symbol is None:
            self.reset_machine()
            return
        if self.current_state == StateMachine.IDLE:
            self.last_symbol = symbol
            self.preamb_counter = 1
            self.current_state = StateMachine.PREAMB
            return
        if self.current_state == StateMachine.PREAMB:
            if self.last_symbol == symbol:
                self.preamb_counter += 1
                if self.preamb_counter == self.preamb_size:
                    self.reference_symbol = symbol
                    self.result_length = 0
                    self.length_counter = 0
                    self.current_state = StateMachine.LENG
                return
            self.reset_machine(symbol)
            return
        if self.current_state == StateMachine.LENG:
            current_step = self.length_counter * self.spread_factor
            self.result_length += ((symbol - self.reference_symbol) % 2**self.spread_factor) << current_step
            self.length_counter += 1
            if self.length_counter == self.leng_size:
                if self.result_length == 0:
                    self.reset_machine(symbol)
                    return
                self.result_length *= self.leng_size
                self.data_counter = 0
                self.result_symbols = []
                self.current_state = StateMachine.DATA
                return
            return
        if self.current_state == StateMachine.DATA:
            self.result_symbols.append((symbol - self.reference_symbol) % 2**self.spread_factor)
            self.data_counter += 1
            if self.data_counter == self.result_length:
                value = self.symbols2string(self.result_symbols)
                self.package = PackageInfo(
                    status=PackageState.PACKAGE_DETECTED,
                    value=value,
                    checksum=self.crc_calculator.checksum(bytes(value, "UTF-8")),
                )
                self.checksum = 0
                self.checksum_counter = 0
                self.current_state = StateMachine.CHECKSUM
            return
        if self.current_state == StateMachine.CHECKSUM:
            current_step = self.checksum_counter * self.spread_factor
            self.checksum += ((symbol - self.reference_symbol) % 2**self.spread_factor) << current_step
            self.checksum_counter += 1
            if self.checksum_counter == self.checksum_size:
                if self.checksum == self.package.checksum:
                    self.package.status = PackageState.PACKAGE_READY
                    self.last_package = self.package
                else:
                    self.package.status = PackageState.CRC_ERROR
                self.current_state = StateMachine.SKIP_NEXT
            return

        if self.current_state == StateMachine.SKIP_NEXT:
            self.reset_machine(symbol)
