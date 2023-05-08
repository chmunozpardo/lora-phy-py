"""Module with an audio reception LoRa demodulator"""
import numpy as np
import pyaudio

from demodulator import Demodulator, StateMachine


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

    # Demodulator
    demod = Demodulator(samples, sample_rate, initial_freq, end_freq, spreading_factor, redundancy)

    # PyAudio callback function
    def callback(in_data, frame_count, time_info, flag):  # pylint: disable=unused-argument
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        symbol = demod.detect_symbol(audio_data)
        demod.state_machine(symbol)
        print(demod.current_state, demod.result_length)
        if demod.current_state == StateMachine.SKIP_NEXT:
            print(demod.package)
        return (audio_data, pyaudio.paContinue)

    # Create PyAudio instance
    py_audio = pyaudio.PyAudio()
    stream = py_audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=samples,
        stream_callback=callback,
    )

    # Start stream and let it process data
    stream.start_stream()
    while stream.is_active():
        pass

    # Stop stream and close everything
    stream.stop_stream()
    stream.close()
    py_audio.terminate()


# Run main function
if __name__ == "__main__":
    main()
