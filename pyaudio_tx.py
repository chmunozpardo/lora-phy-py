import numpy as np
from modulator import Modulator
import pyaudio

# Main process
def main():
    # LoRa Parameters
    Fs = 48000  # Sample rate [samples/s]
    BW = Fs // 2  # Bandwidth [Hz]
    fa = 0  # Start frequency [Hz]
    fb = BW  # Stop frequency [Hz]
    SF = 4  # Spread factor == Bits per symbol
    samples = 16384  # Number of samples

    # Modulator and demodulator
    mod = Modulator(samples, Fs, fa, fb, SF)

    # Preamble
    preamble = [0] * 8

    # String to transmit
    stringData = "Hola"
    data = mod.stringToSymbols(stringData)
    preamble = np.concatenate((preamble, data))
    y_preamble: np.ndarray = mod.getSignal(preamble)

    # Append null at start
    null_start = np.zeros(samples)
    y_preamble = np.concatenate((null_start, y_preamble))

    # Generate white noise
    noise_preamble = np.random.normal(0, 2.0, size=len(y_preamble))

    # Calculate signal, noise power and SNR
    timeLength = 1 / len(y_preamble)
    sign = 10 * np.log10(np.sum(y_preamble**2) * timeLength)
    noise = 10 * np.log10(np.sum(noise_preamble**2) * timeLength)
    print(
        "Noise Power: %.3f - Signal Power: %.3f - SNR: %.3f"
        % (noise, sign, sign - noise),
    )

    # Add noise
    y_preamble += noise_preamble

    # Create PyAudio instance
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=Fs,
        output=True,
    )

    # Write data to stream
    stream.write(y_preamble.astype(np.float32).tobytes())

    # Stop stream and close everything
    stream.stop_stream()
    stream.close()
    pa.terminate()


# Run main function
main()
