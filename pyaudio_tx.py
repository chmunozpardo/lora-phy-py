"""Module with an audio reception LoRa modulator"""
import numpy as np
from numpy.typing import NDArray
import pyaudio

from modulator import Modulator


# Main process
def main():
    # LoRa Parameters
    sample_rate = 48000  # Sample rate [samples/s]
    bandwidth = sample_rate // 2  # Bandwidth [Hz]
    initial_freq = 0  # Start frequency [Hz]
    end_freq = bandwidth  # Stop frequency [Hz]
    spreading_factor = 4  # Spread factor == Bits per symbol
    samples = 16384  # Number of samples
    redundancy = 4  # Redundancy

    # Modulator
    mod = Modulator(samples, sample_rate, initial_freq, end_freq, spreading_factor, redundancy)

    # Preamble
    preamble = [0] * 8

    # String to transmit
    string_data = "Hola"
    data = mod.str2symbols(string_data)
    preamble = np.concatenate((preamble, data))
    y_preamble = mod.generate_signal(preamble)

    # Append null at start
    null_start = np.zeros(samples)
    y_preamble: NDArray[np.float64] = np.concatenate((null_start, y_preamble))

    # Generate white noise
    noise_value = 1.0
    noise_preamble = np.random.normal(0, noise_value, size=len(y_preamble))

    # Calculate signal, noise power and SNR
    time_length = 1 / len(y_preamble)
    sign = 10 * np.log10(np.sum(y_preamble**2) * time_length)
    noise = 10 * np.log10(np.sum(noise_preamble**2) * time_length)
    print(f"Noise Power: {noise:.3} - Signal Power: {sign:.3} - SNR: {sign-noise:.3}")

    # Add noise
    y_preamble += noise_preamble

    # Create PyAudio instance
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate * redundancy,
        output=True,
    )

    # Write data to stream
    stream.write(y_preamble.astype(np.float32).tobytes())

    # Stop stream and close everything
    stream.stop_stream()
    stream.close()
    py_audio.terminate()


# Run main function
if __name__ == "__main__":
    main()
