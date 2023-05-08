# LoRa PHY Mod
This code is a modification of the LoRa PHY layer.

## Modulator and demodulator

The modulator and demodulator uses the following arguments:
```python
# LoRa Parameters
sample_rate: int # Sample rate [samples/s]
initial_freq: float  # Start frequency [Hz]
end_freq: float  # Stop frequency [Hz]
spreading_factor: int  # Spread factor == Bits per symbol
samples: int  # Number of samples
redundancy: int  # Redundancy. This is used to improve SNR by repeating the chirps this number of times for one symbol
```
Common use is as follows:
### Modulator
```python
# Modulator
mod = Modulator(samples, sample_rate, initial_freq, end_freq, spreading_factor, redundancy)

# Preamble used to detect incoming package
preamble = [0] * 8

# Data to transmit
string_data = "Hola"
# Encode data as symbols
data = mod.str2symbols(string_data)
# Concatenate preamble and symbols
preamble = np.concatenate((preamble, data))
# Signal to be transmitted
y_preamble = mod.generate_signal(preamble)
```

### Demodulator
```python
# Detect symbol in signal. Signal must contain the same samples as defined in the Demodulator
symbol = demod.detect_symbol(signal)
# Move Demodulator to the next state depending of detected symbol
demod.state_machine(symbol)
# SKIP_NEXT is the last state of the state machine. If has been reached, check the content.
if demod.current_state == StateMachine.SKIP_NEXT:
    # self.package contains the data, checksum and if the package was correctly received.
    print(demod.package)
```

## Tests
The script `single_test.py` can be used to test different parameters and modifications to the code.