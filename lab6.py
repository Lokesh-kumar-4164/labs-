# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq

fs, N = 256, 2560
t = np.linspace(0, 10, N, endpoint=False)

bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 40)
}

colors = ['red','green','blue','yellow','pink']

np.random.seed(42)
raw = (2*np.sin(2*np.pi*2*t) + 
       1.5*np.sin(2*np.pi*6*t) +
       3*np.sin(2*np.pi*10*t) + 
       np.sin(2*np.pi*20*t) +
       0.5*np.random.randn(N))

raw[500:550] += 80  

def bpf(data, lo, hi):
    b, a = signal.butter(4, [lo/(fs/2), hi/(fs/2)], btype='band')
    return signal.filtfilt(b, a, data)

filtered = bpf(raw, 0.5, 40)

b_notch, a_notch = signal.iirnotch(50, 30, fs)
clean = signal.filtfilt(b_notch, a_notch, filtered)


clean[np.abs(clean) > 75] = 0


freqs = rfftfreq(N, 1/fs)
fft_vals = rfft(clean)

psd = (1/N) * np.abs(fft_vals)**2

power = {}
for band, (lo, hi) in bands.items():
    mask = (freqs >= lo) & (freqs <= hi)
    power[band] = np.trapezoid(psd[mask], freqs[mask])


fig, ax = plt.subplots(4, 1, figsize=(12, 10))
fig.suptitle('EEG Analysis - Lab 6')

ax[0].plot(t, raw, color='gray', alpha=0.5, label='Raw')
ax[0].plot(t, clean, color='steelblue', label='Filtered')
ax[0].axvspan(500/fs, 550/fs, color='red', alpha=0.2, label='Blink')
ax[0].set(title='Raw vs Preprocessed', ylabel='µV', xlabel='Time (s)')
ax[0].legend(fontsize=8)


for i, ((name, (lo, hi)), col) in enumerate(zip(bands.items(), colors)):
    band_signal = bpf(clean, lo, hi)
    ax[1].plot(t, band_signal + i*10, color=col, label=name, lw=0.9)

ax[1].set(title='Brain Waves', ylabel='µV', xlabel='Time (s)')
ax[1].legend(fontsize=8)


ax[2].semilogy(freqs, psd, color='steelblue', lw=1)

for (lo, hi), col in zip(bands.values(), colors):
    mask = (freqs >= lo) & (freqs <= hi)
    ax[2].fill_between(freqs[mask], psd[mask], alpha=0.3, color=col)

ax[2].set(title='Power Spectral Density', xlabel='Hz', ylabel='Power')
ax[2].set_xlim([0, 45])


ax[3].bar(power.keys(), power.values(), color=colors)
ax[3].set(title='Band Power Summary', ylabel='µV²')

plt.tight_layout()
plt.show()


print("\nBand Powers:")
for b, p in power.items():
    print(f"  {b}: {p:.2f} µV²")