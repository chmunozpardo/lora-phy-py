import numpy as np
from demodulator import Demodulator, StateMachine
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

    # Demodulator
    demod = Demodulator(samples, Fs, fa, fb, SF)

    # PyAudio callback function
    def callback(in_data, frame_count, time_info, flag):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        symbol = demod.detectSymbol(audio_data)
        demod.stateMachine(symbol)
        print(demod.currentState, demod.resultLength)
        if demod.currentState == StateMachine.SKIP_NEXT:
            print(demod.package)
        return (audio_data, pyaudio.paContinue)

    # Create PyAudio instance
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=Fs,
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
    pa.terminate()


# Run main function
main()
