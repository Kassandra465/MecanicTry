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

n_modes = 6          # Nombre de modes considérés

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

# --- PARAMETERS ---
A_force = 500.0      # [N]
f_exc = 2.4          # [Hz]
Omega = 2 * np.pi * f_exc
fs = 200.0           # sampling rate [Hz]
t_end = 5.0          # duration [s]
dt = 1 / fs
t = np.arange(0, t_end, dt)

# ==========================
# Exact stationary response (FRF method)
# ==========================
def FRF_solution(M, K, F, Omega):
    A = -Omega**2 * M + K
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

# ==========================
# PSD and RMS
# ==========================
def compute_PSD_and_RMS(signal, fs):
    N = len(signal)
    T = N / fs  # total duration
    Y = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    # Keep positive frequencies only
    pos_idx = freqs >= 0
    f_pos = freqs[pos_idx]
    PSD = 2 * (np.abs(Y[pos_idx]) ** 2) / T  # (m/s²)²/Hz

    # Compute RMS
    df = f_pos[1] - f_pos[0]
    integrale = np.sum(PSD) * df
    rms = np.sqrt(integrale)

    return f_pos, PSD, rms

# ==========================
# Convergence study
# ==========================
def convergence_study(Phi, M, K, F, omega_n, Omega, U_ref, idx, n_modes):
    amp_ref = abs(U_ref[idx])
    amp_disp, amp_acc = [], []

    # Initialize cumulative modal responses
    u_disp_sum = np.zeros_like(U_ref, dtype=complex)
    u_acc_sum = np.zeros_like(U_ref, dtype=complex)

    # Loop over modes incrementally
    for k in range(1, n_modes + 1):
        phi_k = Phi[:, k-1]
        mu_k = phi_k.T @ M @ phi_k
        denom = (omega_n[k-1]**2 - Omega**2)

        # Modal displacement contribution
        qk_disp = (phi_k.T @ F) / (mu_k * denom)
        u_disp_sum += phi_k * qk_disp

        # Modal acceleration contribution (with static correction)
        qk_acc = (phi_k.T @ F) / (mu_k * (omega_n[k-1]**2 - Omega**2) * (omega_n[k-1]**2))
        u_acc_sum += phi_k * qk_acc

        # Store amplitude for current number of modes
        amp_disp.append(abs(u_disp_sum[idx]))
        amp_acc.append(abs((np.linalg.inv(K) @ F + (Omega**2) * u_acc_sum)[idx]))

    # Convert to numpy arrays
    amp_disp = np.array(amp_disp)
    amp_acc = np.array(amp_acc)

    # Relative errors (%)
    err_disp = 100 * np.abs((amp_disp - amp_ref) / amp_ref)
    err_acc = 100 * np.abs((amp_acc - amp_ref) / amp_ref)

    return amp_ref, amp_disp, amp_acc, err_disp, err_acc

# ==========================
# Main
# ==========================
def main2():
    # Temps (steady-state)
    dt = 1 / fs
    t = np.arange(0, t_end, dt)
    # Réponses stationnaires
    U_ref = FRF_solution(M_ass, K_ass, F, Omega)
    u_time = np.real(np.exp(1j * Omega * t)[:, None] * U_ref[None, :])
    u_exc = u_time[:, dof_y]
    x_disp = modal_displacement(Phi, M_ass, F, omega_n, Omega, n_modes)
    x_disp_corrected = np.linalg.inv(K_ass) @ F + x_disp
    U_modal = modal_displacement(Phi, M_ass, F, omega_n, Omega, n_modes)
    u_modal = np.real(np.exp(1j * Omega * t)[:, None] * U_modal[None, :])
    u_modal_exc = u_modal[:, dof_y]

    print("Amp FRF:", abs(U_ref[idx])),
    print("Amp disp", abs(x_disp[idx]))
    print("Amp disp-corrected", abs(x_disp_corrected[idx]))
    x_acc = modal_acceleration(Phi, M_ass, K_ass, F, omega_n, Omega, n_modes)
    print("Amp acc:", abs(x_acc[idx]))


    u_exc = np.real(U_ref[idx] * np.cos(Omega * t))
    # Modal displacement & acceleration
    u_disp = np.real(modal_displacement(Phi, M_ass, F, omega_n, Omega, n_modes)[idx] * np.cos(Omega * t))
    u_acc = np.real(modal_acceleration(Phi, M_ass, K_ass, F, omega_n, Omega, n_modes)[idx] * np.cos(Omega * t))

    # FFT
    f_ref, fft_ref = FFT_signal(u_exc, dt)
    f_md, fft_md = FFT_signal(u_disp, dt)
    f_ma, fft_ma = FFT_signal(u_acc, dt)

    acc_exc = - (Omega ** 2) * u_exc
    f_psd, PSD, rms = compute_PSD_and_RMS(acc_exc, fs)
    print(f"RMS : {rms:.3e}")

    #convergence
    amp_ref, amp_disp, amp_acc, err_disp, err_acc = convergence_study(Phi, M_ass, K_ass, F, omega_n, Omega, U_ref, idx, n_modes)

    # Plots steady-state response
    # Exact FRF response
    plt.figure(figsize=(8, 4))
    plt.plot(t, u_exc, 'k', linewidth=1.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title('Exact steady-state response (FRF)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Modal displacement method
    plt.figure(figsize=(8, 4))
    plt.plot(t, u_disp, 'b', linewidth=1.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title('Steady-state response - Modal Displacement Method')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Modal acceleration method
    plt.figure(figsize=(8, 4))
    plt.plot(t, u_acc, 'r', linewidth=1.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title('Steady-state response - Modal Acceleration Method')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #combined plot
    plt.figure()
    plt.plot(t, u_exc, '--', label='Exact FRF')
    plt.plot(t, u_disp, label='Modal displacement')
    plt.plot(t, u_acc, ':', label='Modal acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.legend(); plt.grid(); plt.title('Steady-state response'); plt.show()

    #plot FFT
    plt.figure()
    plt.plot(f_ref, fft_ref, label='FRF')
    plt.plot(f_md, fft_md, label='Modal displacement')
    plt.plot(f_ma, fft_ma, label='Modal acceleration')
    plt.xlim(0, 10); plt.legend(); plt.grid()
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Amplitude')
    plt.title('FFT at the excitation node'); plt.show()

    #Plot convergence
    modes = np.arange(1, n_modes + 1)
    plt.figure(figsize=(8,5))
    plt.plot(modes, amp_disp, 'o-', label='Modal displacement')
    plt.plot(modes, amp_acc, 's--', label='Modal acceleration')
    plt.hlines(amp_ref, 1, n_modes, colors='k', linestyles=':', label='FRF reference')
    plt.yscale('log')
    plt.xlabel('Number of modes')
    plt.ylabel('Amplitude [m]')
    plt.title('Convergence of modal approximations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot relative errors
    plt.figure(figsize=(8,5))
    plt.plot(modes, err_disp, 'o-', label='Rel. error displacement [%]')
    plt.plot(modes, err_acc, 's--', label='Rel. error acceleration [%]')
    plt.yscale('log')
    plt.xlabel('Number of modes')
    plt.ylabel('Relative error (%)')
    plt.title('Relative error vs number of modes')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    #plot PSD
    plt.figure()
    plt.semilogy(f_psd, PSD)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [(m/s²)²/Hz]')
    plt.title('PSD of the lateral acceleration')
    plt.grid(); plt.show()

    #Resulats
    print("\nMode |   amp_disp [m]    |    amp_acc [m]    | abs_err_disp [m] | abs_err_acc [m]")
    for k in range(n_modes):
        print(f"{k + 1:9d} | {amp_disp[k]:12.4e} | {amp_acc[k]:12.4e} | "
              f"{abs(amp_disp[k] - amp_ref):14.4e} | {abs(amp_acc[k] - amp_ref):14.4e}")

    print("\n----- SUMMARY -----")
    print(f"A = {A_force} N, f = {f_exc} Hz, Ω = {Omega:.3f} rad/s")
    print(f"FRF amplitude (magnitude): {amp_ref:.3e} m")
    print("Relative errors (%) - displacement:", err_disp)
    print("Relative errors (%) - acceleration:", err_acc)


if __name__ == "__main__":
    main2()
