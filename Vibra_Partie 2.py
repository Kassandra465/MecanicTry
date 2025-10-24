import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from numpy.linalg import inv
from scipy.signal.windows import cosine

from Vibra_Partie1 import M_ass, K_ass, frequencies, eigvecs, dofList

# PARAMÈTRES
A_force = 500.0       # [N] Amplitude de la force
f_exc = 2.4           # [Hz] Fréquence d’excitation
Omega = 2 * np.pi * f_exc
fs = 200              # [Hz] Fréquence d’échantillonnage
t_end = 5.0           # [s] Durée d’observation
zeta_target = 0.005   # Amortissement 0.5%

n_modes = 6           # Nombre de modes considérés

# FORCE D’EXCITATION
node_green = 6
dof_y = dofList[node_green - 1][1]     # DOF en direction Y
idx = int(dof_y) - 1                   # passage en index 0-based

F = np.zeros(M_ass.shape[0])
F[idx] = A_force

#FRÉQUENCES ET MODES PROPRES
freqs = np.array(frequencies)
omega_n = 2 * np.pi * freqs
Phi = np.array(eigvecs)
n_modes = min(Phi.shape[1], n_modes)

# ==========================
# Rayleigh damping
# ==========================-
def rayleigh_damping(M, K, omega, zeta=zeta_target):
    w1, w2 = omega[0], omega[1]
    beta = (2 * zeta) / (w1 + w2)
    alpha = beta * w1 * w2
    C = alpha * M + beta * K
    zetas = [0.5 * (alpha / w + beta * w) for w in omega[:n_modes]]
    return C, alpha, beta, np.array(zetas)

# ==========================
# Exact stationary response (FRF method)
# ==========================
def FRF_solution(M, C, K, F, Omega):
    A = -Omega**2 * M + 1j * Omega * C + K
    return inv(A) @ F

# ==========================
# Modal displacement method
# ==========================
def modal_displacement(Phi, M, F, omega_n, Omega, n_modes):
    nd = M.shape[0]
    u_t = np.zeros((nd,), dtype=float)
    for r in range(n_modes):
        x_r = Phi[:, r]
        mu_r = x_r.T @ M @ x_r
        num = x_r * (x_r.T @ F)
        denom = (omega_n[r]**2 - Omega**2) * mu_r
        u_t += np.real(num / denom)
    return u_t

# ==========================
# Modal acceleration method
# ==========================
def modal_acceleration(Phi, M, K, F, omega_n, Omega, n_modes):
    nd = M.shape[0]
    K_inv = np.linalg.inv(K)
    term = np.zeros((nd,), dtype=float)
    for r in range(n_modes):
        x_r = Phi[:, r]
        mu_r = x_r.T @ M @ x_r
        num = x_r * (x_r.T @ F)
        denom = (omega_n[r]**2 - Omega**2) * (omega_n[r]**2) * mu_r
        term += np.real(num / denom)
    u_t = (K_inv @ F) + (Omega**2) * term
    return u_t

# ==========================
# FFT (discrete Fourier transform)
# ==========================
def FFT_signal(y, dt):
    N = len(y)
    Y = fft(y)
    f = fftfreq(N, dt)[:N // 2]
    mag = 2 / N * np.abs(Y[:N // 2])
    return f, mag

def compute_PSD_and_RMS(signal, fs):
    # Duration of the signal
    T = len(signal) / fs

    # FFT computation
    Y = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    # Keep only positive frequencies
    pos_idx = freqs >= 0
    freqs = freqs[pos_idx]
    Y = Y[pos_idx]

    # PSD according to definition: |Y(f)|^2 / T
    PSD = (np.abs(Y)**2) / T

    # RMS computed as sqrt(integral(PSD df))
    df = freqs[1] - freqs[0]
    rms_value = np.sqrt(np.trapezoid(PSD, freqs))

    return freqs, PSD, rms_value

# ==========================
# Main
# ==========================
def main2():
    # Amortissement
    C, alpha, beta, zetas = rayleigh_damping(M_ass, K_ass, omega_n)
    print(f"\nRayleigh damping: α={alpha:.3e}, β={beta:.3e}")
    for i, z in enumerate(zetas, 1):
        print(f"Mode {i}: f = {freqs[i-1]:.3f} Hz, ζ = {z:.4f}")

    # Réponses stationnaires
    U_ref = FRF_solution(M_ass, C, K_ass, F, Omega)
    x_disp = modal_displacement(Phi, M_ass, F, omega_n, Omega, n_modes)
    print("Amp FRF:", abs(U_ref[idx])),
    print("Amp disp", abs(x_disp[idx]))
    x_acc = modal_acceleration(Phi, M_ass, K_ass, F, omega_n, Omega, n_modes)
    print("Amp acc:", abs(x_acc[idx]))

    # Time reconstruction (steady-state)
    dt = 1 / fs
    t = np.arange(0, t_end, dt)

    u_exc = np.real(U_ref[idx] * np.cos(Omega * t))
    # Modal displacement & acceleration approximations
    u_disp = np.real(modal_displacement(Phi, M_ass, F, omega_n, Omega, n_modes)[idx] * np.cos(Omega * t))
    u_acc = np.real(modal_acceleration(Phi, M_ass, K_ass, F, omega_n, Omega, n_modes)[idx] * np.cos(Omega * t))

    # FFT
    f_ref, fft_ref = FFT_signal(u_exc, dt)
    f_md, fft_md = FFT_signal(u_disp, dt)
    f_ma, fft_ma = FFT_signal(u_acc, dt)

    f_psd, PSD, rms = compute_PSD_and_RMS(u_exc, fs)
    print(f"RMS (computed): {rms:.3e}")

    # Convergence
    amp_ref = abs(U_ref[idx])
    amp_disp, amp_acc = [], []
    for k in range(1, n_modes + 1):
        xd = modal_displacement(Phi, M_ass, F, omega_n, Omega, k)
        x_d = np.linalg.inv(K_ass) @ F + xd
        xa = modal_acceleration(Phi, M_ass, K_ass, F, omega_n, Omega, k)
        amp_disp.append(abs(x_d[idx]))
        amp_acc.append(abs(xa[idx]))

    err_disp = 100 * abs((amp_disp - amp_ref) / amp_ref)
    err_acc = 100 * abs((amp_acc - amp_ref) / amp_ref)

    amp_ref = abs(U_ref[idx])
    print("Mode count | amp_disp [m]    | amp_acc [m]    | abs_err_disp [m] | abs_err_acc [m]")
    for k in range(1, n_modes + 1):
        xd = modal_displacement(Phi, M_ass, F, omega_n, Omega, k)
        xa = modal_acceleration(Phi, M_ass, K_ass, F, omega_n, Omega, k)
        print(
            f"{k:9d} | {abs(xd[idx]):12.4e} | {abs(xa[idx]):12.4e} | {abs(abs(xd[idx]) - amp_ref):14.4e} | {abs(abs(xa[idx]) - amp_ref):14.4e}")

    # PLOTS AND RESULTS
    plt.figure()
    plt.plot(t, u_exc, label='Exact FRF')
    plt.plot(t, u_disp, '--', label='Modal displacement')
    plt.plot(t, u_acc, ':', label='Modal acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.legend(); plt.grid(); plt.title('Steady-state response'); plt.show()

    plt.figure()
    plt.plot(f_ref, fft_ref, label='FRF')
    plt.plot(f_md, fft_md, label='Modal displacement')
    plt.plot(f_ma, fft_ma, label='Modal acceleration')
    plt.xlim(0, 10); plt.legend(); plt.grid()
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Amplitude')
    plt.title('FFT at the excitation node'); plt.show()

    plt.figure()
    plt.semilogy(f_psd, PSD)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [(m/s²)²/Hz]')
    plt.title('PSD of the lateral acceleration')
    plt.grid(); plt.show()


    modes = np.arange(1, n_modes + 1)
    plt.figure()
    plt.plot(modes, amp_disp, 'o-', label='Displacement method')
    plt.plot(modes, amp_acc, 's--', label='Acceleration method')
    plt.hlines(amp_ref, 1, n_modes, colors='k', linestyles=':', label='FRF reference')
    plt.xlabel('Number of modes'); plt.ylabel('Amplitude [m]')
    plt.legend(); plt.grid(); plt.title('Convergence of modal methods'); plt.show()

    print("\n----- SUMMARY -----")
    print(f"A = {A_force} N, f = {f_exc} Hz, Ω = {Omega:.3f} rad/s")
    print(f"FRF amplitude (magnitude): {amp_ref:.3e} m")
    print("Relative errors (%) - displacement:", err_disp)
    print("Relative errors (%) - acceleration:", err_acc)

# -------------------- EXÉCUTION --------------------
if __name__ == "__main__":
    main2()
