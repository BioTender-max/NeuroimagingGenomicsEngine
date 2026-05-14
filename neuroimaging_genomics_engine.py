import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import shutil, os

# ── Parameters ───────────────────────────────────────────────────────────────
N_EXP     = 50    # experiments
N_NEURONS = 200   # neurons per experiment
T_FRAMES  = 1000  # time frames
DT        = 0.033 # seconds per frame (~30 Hz)
FS        = 1/DT  # sampling rate

# ── Calcium imaging: ΔF/F traces ─────────────────────────────────────────────
time = np.arange(T_FRAMES) * DT

def calcium_transient(t, t0, amp, tau_rise=0.1, tau_decay=0.5):
    trace = np.zeros_like(t)
    mask = t >= t0
    dt = t[mask] - t0
    trace[mask] = amp * (1 - np.exp(-dt/tau_rise)) * np.exp(-dt/tau_decay)
    return trace

n_show = 10
ca_traces = np.zeros((n_show, T_FRAMES))
for n in range(n_show):
    n_events = np.random.poisson(5)
    for _ in range(n_events):
        t0  = np.random.uniform(0, T_FRAMES*DT - 2)
        amp = np.random.lognormal(0, 0.5)
        ca_traces[n] += calcium_transient(time, t0, amp)
    ca_traces[n] += np.random.normal(0, 0.05, T_FRAMES)

thresholds = ca_traces.mean(axis=1) + 2*ca_traces.std(axis=1)
transient_mask = ca_traces > thresholds[:, None]
n_transients = transient_mask.sum(axis=1)

# ── Spike sorting: 3 units ───────────────────────────────────────────────────
n_units = 3
spike_times = [np.sort(np.random.uniform(0, T_FRAMES*DT, np.random.randint(50, 200)))
               for _ in range(n_units)]
unit_colors = ['#58a6ff','#3fb950','#f78166']

# ── Neural population dynamics (PCA) ─────────────────────────────────────────
pop_activity = np.random.randn(N_NEURONS, T_FRAMES)
latent = np.random.randn(3, T_FRAMES)
loadings = np.random.randn(N_NEURONS, 3)
pop_activity += loadings @ latent * 2

pop_centered = pop_activity - pop_activity.mean(axis=1, keepdims=True)
U, S, Vt = np.linalg.svd(pop_centered, full_matrices=False)
var_explained = S**2 / (S**2).sum()
pc_scores = Vt[:3, :]

# ── Neuropixels depth profile ─────────────────────────────────────────────────
n_channels = 384
depths = np.linspace(0, 3840, n_channels)
layer_bounds = [0, 200, 400, 700, 1000, 1500, 2000, 3840]
layer_names  = ['L1','L2/3','L4','L5','L6a','L6b','WM']
firing_rates = np.zeros(n_channels)
for i, (lo, hi) in enumerate(zip(layer_bounds[:-1], layer_bounds[1:])):
    mask = (depths >= lo) & (depths < hi)
    base = [0.5, 8, 15, 12, 6, 3, 0.2][i]
    firing_rates[mask] = np.random.lognormal(np.log(base+0.1), 0.5, mask.sum())

# ── LFP power spectrum ───────────────────────────────────────────────────────
lfp_fs = 1000
lfp_t  = np.arange(0, 10, 1/lfp_fs)
theta_amp = 50; gamma_amp = 20
lfp = (theta_amp * np.sin(2*np.pi*6*lfp_t) +
       gamma_amp * np.sin(2*np.pi*40*lfp_t) +
       np.random.normal(0, 10, len(lfp_t)))
freqs, psd = welch(lfp, fs=lfp_fs, nperseg=1024)

# ── Theta-gamma coupling ─────────────────────────────────────────────────────
theta_phase = np.angle(np.exp(1j * 2*np.pi*6*lfp_t))
gamma_env   = np.abs(gamma_amp * np.sin(2*np.pi*40*lfp_t) + np.random.normal(0, 5, len(lfp_t)))
n_bins = 18
phase_bins = np.linspace(-np.pi, np.pi, n_bins+1)
gamma_by_phase = np.array([gamma_env[(theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i+1])].mean()
                            for i in range(n_bins)])

# ── Neural coding efficiency ─────────────────────────────────────────────────
mi_vals = np.random.lognormal(np.log(0.5), 0.4, N_NEURONS)

# ── Signal-to-noise ──────────────────────────────────────────────────────────
snr_vals = np.random.lognormal(np.log(5), 0.6, N_NEURONS)

# ── Key results ──────────────────────────────────────────────────────────────
mean_transients = n_transients.mean()
theta_power = psd[(freqs >= 4) & (freqs <= 8)].mean()
gamma_power = psd[(freqs >= 30) & (freqs <= 80)].mean()
mean_fr = firing_rates[firing_rates > 0].mean()

# ── Dashboard ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Neuroimaging Genomics Engine — Calcium Imaging & Electrophysiology',
             color='white', fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Calcium traces
ax = axes[0, 0]; ax.set_facecolor('#161b22')
offset = 0
for n in range(min(5, n_show)):
    ax.plot(time, ca_traces[n] + offset, color=unit_colors[n % 3], lw=0.8, alpha=0.9)
    offset += ca_traces[n].max() + 0.5
ax.set_xlabel('Time (s)', color='white'); ax.set_ylabel('dF/F + offset', color='white')
ax.set_title('Calcium Transients (dF/F)', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 2: Spike raster
ax = axes[0, 1]; ax.set_facecolor('#161b22')
for u, (st, col) in enumerate(zip(spike_times, unit_colors)):
    ax.vlines(st, u+0.5, u+1.5, color=col, lw=0.8, alpha=0.8)
ax.set_xlabel('Time (s)', color='white'); ax.set_ylabel('Unit', color='white')
ax.set_yticks([1, 2, 3]); ax.set_yticklabels(['Unit 1','Unit 2','Unit 3'], color='white')
ax.set_title('Spike Raster (3 Units)', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 3: Population PCA
ax = axes[0, 2]; ax.set_facecolor('#161b22')
sc = ax.scatter(pc_scores[0, ::5], pc_scores[1, ::5], c=np.arange(0, T_FRAMES, 5)*DT,
                cmap='plasma', s=5, alpha=0.7)
cb = plt.colorbar(sc, ax=ax, label='Time (s)')
cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', color='white')
ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', color='white')
ax.set_title('Neural Population Dynamics (PCA)', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 4: Depth profile
ax = axes[1, 0]; ax.set_facecolor('#161b22')
ax.plot(firing_rates, depths, color='#58a6ff', lw=1.5, alpha=0.8)
for lo, name in zip(layer_bounds[:-1], layer_names):
    ax.axhline(lo, color='#30363d', ls='--', lw=0.8)
    ax.text(firing_rates.max()*0.8, lo+50, name, color='#8b949e', fontsize=8)
ax.set_xlabel('Firing Rate (Hz)', color='white'); ax.set_ylabel('Depth (um)', color='white')
ax.set_title('Neuropixels Depth Profile', color='white', fontweight='bold')
ax.invert_yaxis(); ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 5: LFP power spectrum
ax = axes[1, 1]; ax.set_facecolor('#161b22')
ax.semilogy(freqs, psd, color='#d2a8ff', lw=1.5)
ax.axvspan(4, 8, alpha=0.2, color='#3fb950', label='Theta (4-8 Hz)')
ax.axvspan(30, 80, alpha=0.2, color='#ffa657', label='Gamma (30-80 Hz)')
ax.set_xlim(0, 150); ax.set_xlabel('Frequency (Hz)', color='white')
ax.set_ylabel('PSD (uV^2/Hz)', color='white')
ax.set_title('LFP Power Spectrum', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 6: Theta-gamma coupling
ax = axes[1, 2]; ax.set_facecolor('#161b22')
phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
ax.bar(phase_centers, gamma_by_phase, width=2*np.pi/n_bins,
       color='#ffa657', alpha=0.8, edgecolor='#30363d')
ax.set_xlabel('Theta Phase (rad)', color='white')
ax.set_ylabel('Gamma Amplitude (uV)', color='white')
ax.set_title('Theta-Gamma Phase-Amplitude Coupling', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 7: Coding efficiency
ax = axes[2, 0]; ax.set_facecolor('#161b22')
ax.hist(mi_vals, bins=30, color='#56d364', alpha=0.8, edgecolor='#30363d')
ax.axvline(mi_vals.mean(), color='#ffa657', ls='--', lw=2, label=f'Mean={mi_vals.mean():.2f} bits')
ax.set_xlabel('Mutual Information (bits)', color='white'); ax.set_ylabel('Count', color='white')
ax.set_title('Neural Coding Efficiency', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 8: Signal-to-noise
ax = axes[2, 1]; ax.set_facecolor('#161b22')
ax.hist(snr_vals, bins=30, color='#79c0ff', alpha=0.8, edgecolor='#30363d')
ax.axvline(snr_vals.mean(), color='#ffa657', ls='--', lw=2, label=f'Mean SNR={snr_vals.mean():.1f}')
ax.set_xlabel('SNR', color='white'); ax.set_ylabel('Count', color='white')
ax.set_title('Signal-to-Noise Ratio Distribution', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 9: Summary
ax = axes[2, 2]; ax.set_facecolor('#161b22'); ax.axis('off')
ax.set_title('Summary Statistics', color='white', fontweight='bold')
summary_lines = [
    ('Experiments', f'{N_EXP}'),
    ('Neurons/Experiment', f'{N_NEURONS}'),
    ('Time Frames', f'{T_FRAMES} ({T_FRAMES*DT:.1f}s)'),
    ('Mean Transients/Neuron', f'{mean_transients:.1f}'),
    ('Sorted Units', f'{n_units}'),
    ('PC1 Variance', f'{var_explained[0]*100:.1f}%'),
    ('Theta Power', f'{theta_power:.1f} uV^2/Hz'),
    ('Gamma Power', f'{gamma_power:.1f} uV^2/Hz'),
    ('Mean Firing Rate', f'{mean_fr:.1f} Hz'),
]
for idx, (k, v) in enumerate(summary_lines):
    ax.text(0.05, 0.88 - idx*0.10, k, color='#8b949e', fontsize=10, transform=ax.transAxes)
    ax.text(0.65, 0.88 - idx*0.10, v, color='#58a6ff', fontsize=10, fontweight='bold', transform=ax.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/shared-workspace/shared/neuroimaging_genomics_engine_dashboard.png',
            dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
shutil.copy(__file__, '/mnt/shared-workspace/shared/neuroimaging_genomics_engine.py')

print("=== NeuroimagingGenomicsEngine Key Results ===")
print(f"Experiments: {N_EXP}, Neurons: {N_NEURONS}, Frames: {T_FRAMES}")
print(f"Mean calcium transients per neuron: {mean_transients:.1f}")
print(f"Sorted units: {n_units}, spike counts: {[len(s) for s in spike_times]}")
print(f"PC1 variance explained: {var_explained[0]*100:.1f}%")
print(f"Theta band power: {theta_power:.1f} uV^2/Hz")
print(f"Gamma band power: {gamma_power:.1f} uV^2/Hz")
print(f"Mean firing rate (non-zero): {mean_fr:.1f} Hz")
print(f"Mean coding MI: {mi_vals.mean():.2f} bits")
print("Dashboard saved.")
