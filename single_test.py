"""Module with some testing routines"""
import string
import random
import numpy as np

import matplotlib.pyplot as plt
from modulator import Modulator
from demodulator import Demodulator, PackageState


LETTERS = string.ascii_letters


class SingleTest:
    def __init__(self, number_tests: int, spread_factor: int, redundancy: int, display: bool = False):
        self.number_tests = number_tests
        self.display = display

        # LoRa Parameters
        sample_frequency = 48000
        init_freq = 0
        end_freq = sample_frequency // 2
        self.spread_factor = spread_factor
        self.samples = 512

        # Modulator and demodulator
        self.mod = Modulator(self.samples, sample_frequency, init_freq, end_freq, spread_factor, redundancy)
        self.demod = Demodulator(self.samples, sample_frequency, init_freq, end_freq, spread_factor, redundancy)

    def single_ber(self, noise: float):
        ber_result = 0
        none_result = 0
        # Preamble and Data
        if self.spread_factor == 7:
            value = random.randint(0, 2 ** (self.spread_factor - 3) - 1)
            encoded_value = self.mod.encode_int(value)
        else:
            value = random.randint(0, 2**self.spread_factor - 1)
            encoded_value = value
        y_preamble = self.mod.generate_signal([encoded_value])

        # Generate white noise
        noise_preamble = np.random.normal(0, noise, size=len(y_preamble))

        # Calculate signal, noise power and SNR
        time_length = 1 / len(y_preamble)
        sign = 10 * np.log10(np.sum(y_preamble**2) * time_length)
        noise = 10 * np.log10(np.sum(noise_preamble**2) * time_length)
        noise_snr = sign - noise

        # Add noise
        y_preamble += noise_preamble

        # Separate signal in chunks of size "samples"
        leng = int(np.floor(len(y_preamble) / self.demod.samples_symbol))
        for index in range(leng):
            start_ind = self.demod.samples_symbol * index
            end_ind = self.demod.samples_symbol * (index + 1)
            result = self.demod.detect_symbol(y_preamble[start_ind:end_ind])
            if result is not None:
                result = self.demod.decode_int(result)
                # Calculate Hamming distance
                r_temp = (1 << np.arange(self.spread_factor))[:, None]
                ber = np.count_nonzero((result & r_temp) != (value & r_temp))
                # Add to BER result variable
                ber_result += ber
            else:
                none_result += 1
                ber_result += self.spread_factor
        return (noise_snr, ber_result, none_result)

    def run_ber(self, noise_array: list[float]):
        noise_snr_list = []
        noise_ber_list = []

        for index, noise in enumerate(noise_array):
            total_snr = 0
            total_ber = 0
            total_none = 0
            print(f"{index:3d}", end=" - ")
            # Run number of tests
            for _ in range(self.number_tests):
                (noise_snr, ber_result, none_result) = self.single_ber(noise)
                total_snr += noise_snr
                total_ber += ber_result
                total_none += none_result
            current_snr = total_snr / self.number_tests
            current_ber = total_ber / self.spread_factor / self.number_tests
            noise_snr_list.append(current_snr)
            noise_ber_list.append(current_ber)
            print(f"SNR: {current_snr:.4f} dB, BER result: {current_ber:.4f}, None result: {total_none}")

        return (np.array(noise_snr_list), np.array(noise_ber_list))

    def single_tx(self, noise_value: float, package_length: int):
        # Preamble and Data
        preamble = [0] + [1] + [0] * 8

        data_length = package_length
        string_data = "".join(random.choice(LETTERS) for _ in range(data_length))
        data = self.mod.str2symbols(string_data)
        preamble = np.concatenate((preamble, data))
        data_preamble = self.mod.generate_signal(preamble)

        # Add random start
        random_samples = random.randint(1, 4000)
        random_start = np.zeros(random_samples)
        random_samples = random.randint(1, 4000)
        random_stop = np.zeros(random_samples)
        y_preamble = np.concatenate((random_start, data_preamble, random_stop))

        # Create noise to add to the signal
        noise_preamble = self.mod.generate_noise(noise_value, len(y_preamble))
        y_preamble += noise_preamble

        sign = 10 * np.log10(np.sum(data_preamble**2) / len(y_preamble))
        noise = 10 * np.log10(np.sum(noise_preamble**2) / len(y_preamble))
        snr_db = sign - noise
        print(f"Noise Power[dB]: {noise:.3} - Signal Power[dB]: {sign:.3} - SNR[dB]: {snr_db:.3}")

        out_symbols = []

        leng = int(np.floor(len(y_preamble) / self.demod.samples_symbol))
        self.demod.reset_machine()
        for index in range(leng):
            init_ind = index * self.demod.samples_symbol
            end_ind = (index + 1) * self.demod.samples_symbol
            result = self.demod.detect_symbol(y_preamble[init_ind:end_ind])
            self.demod.state_machine(result)
            out_symbols.append(result)

        if self.display:
            print(" - Status:", self.demod.package.status)
            print(" - Sent Data:", string_data, package_length)
            print(" - Package Data:", self.demod.package.value, len(self.demod.package.value))
            print(" - Checksum:", self.demod.package.checksum, self.demod.checksum)

        return self.demod.package

    def run_tx(self):
        crc_fail = 0
        package_fail = 0
        proper_tx = 0
        for i in range(self.number_tests):
            print(f"{i:3d}", end=" - ")
            package_info = self.single_tx(1.0, 10)
            if package_info.status == PackageState.PACKAGE_DETECTED:
                package_fail += 1
            elif package_info.status == PackageState.NO_PACKAGE:
                package_fail += 1
            elif package_info.status == PackageState.CRC_ERROR:
                crc_fail += 1
            elif package_info.status == PackageState.PACKAGE_READY:
                proper_tx += 1

        print(f"CRC fail: {crc_fail / self.number_tests * 100.0:.3f}%")
        print(f"Package fail: {package_fail/ self.number_tests * 100.0:.3f}%")
        print(f"Correct: {proper_tx / self.number_tests * 100.0:.3f}%")


if __name__ == "__main__":
    N = 250

    # Uncomment the following lines to simulate noisy transmissions
    # test = SingleTest(N, 8)
    # test.run_tx()

    # The following lines plot the BER parameter for different parameters
    noise_options = list(np.linspace(0.5, 10.0, 3))
    test2 = SingleTest(number_tests=N, spread_factor=7, redundancy=2)
    (noise_snr2, noise_result2) = test2.run_ber(noise_options)
    plt.plot(noise_snr2, noise_result2 * 100, label="Redundancy=2")
    test4 = SingleTest(number_tests=N, spread_factor=7, redundancy=4)
    (noise_snr4, noise_result4) = test4.run_ber(noise_options)
    plt.plot(noise_snr4, noise_result4 * 100, label="Redundancy=4")
    test7 = SingleTest(number_tests=N, spread_factor=7, redundancy=8)
    (noise_snr7, noise_result7) = test7.run_ber(noise_options)
    plt.plot(noise_snr7, noise_result7 * 100, label="Redundancy=8")
    test8 = SingleTest(number_tests=N, spread_factor=7, redundancy=16)
    (noise_snr8, noise_result8) = test8.run_ber(noise_options)
    plt.plot(noise_snr8, noise_result8 * 100, label="Redundancy=16")
    plt.legend()
    plt.title("BER Test")
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER [%]")
    # plt.yscale("log")
    plt.ylim(0, 100)
    plt.show()
