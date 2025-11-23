import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from numpy.linalg import inv
from scipy.signal.windows import cosine

from Vibra_Partie1 import M_ass, K_ass, frequencies, eigvecs, dofList, dof_map

# PARAMÈTRES
A_force = 500.0       # [N] Amplitude de la force
f_exc = 2.4           # [Hz] Fréquence d’excitation
Omega = 2 * np.pi * f_exc
fs = 200              # [Hz] Fréquence d’échantillonnage
t_end = 5.0           # [s] Durée d’observation
dt = 1 / fs
t = np.arange(0, t_end, dt)

n_modes = 6          # Nombre de modes considérés

#FRÉQUENCES ET MODES PROPRES
freqs = np.array(frequencies)
omega_n = 2 * np.pi * freqs
Phi = np.array(eigvecs)
n_modes = min(Phi.shape[1], n_modes)

# FORCE D’EXCITATION
node_green = 6
dof_y = dofList[node_green][1]     # DOF en direction Y
idx = dof_map[dof_y]

F = np.zeros(M_ass.shape[0])
F[idx] = A_force


# ==========================
# Exact stationary response (FRF method)
# ==========================
def FRF_solution(M, K, F, Omega):
    A_1 = -Omega**2 * M + K
    A = np.linalg.inv(A_1) @ F
    return A

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
#PSD = 2 * (np.abs(Y[pos_idx]) ** 2) / T  # (m/s²)²/Hz

def compute_PSD_RMS(signal, fs):
    N = len(signal)
    dt = 1.0 / fs
    Y = fft(signal)
    freqs = fftfreq(N, d=dt)

    #Garde les fréquences positives
    pos_idx = freqs >= 0
    f_pos = freqs[pos_idx]

    #PSD
    PSD = (2.0 / (N * fs)) * (np.abs(Y[pos_idx]) ** 2)

    #RMS
    df = f_pos[1] - f_pos[0]
    mean_square = np.sum(PSD) * df
    rms = np.sqrt(mean_square) * (Omega ** 2)

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
    # Référence FRF (Exacte)
    U_ref = FRF_solution(M_ass, K_ass, F, Omega)
    amp_ref = U_ref[idx]  # Amplitude exacte au bon DOF

    # Méthode Déplacement
    U_disp = modal_displacement(Phi, M_ass, F, omega_n, Omega, n_modes)
    amp_disp = U_disp[idx]

    U_disp_corrected = np.linalg.inv(K_ass) @ F + U_disp
    amp_disp_corrected = U_disp_corrected[idx]

    # Méthode Accélération
    U_acc = modal_acceleration(Phi, M_ass, K_ass, F, omega_n, Omega, n_modes)
    amp_acc = U_acc[idx]

    # Affichage des amplitudes
    print(f"Index DOF (dof_y): {dof_y}")
    print(f"Amp FRF (exacte): {abs(amp_ref):.5e} m")
    print(f"Amp Disp: {abs(amp_disp):.5e} m")
    print(f"Amp Acc: {abs(amp_acc):.5e} m")

    # Génération des séries temporelles (pour les plots)
    u_exc = np.real(amp_ref * np.cos(Omega * t))
    u_disp = np.real(amp_disp * np.cos(Omega * t))
    u_acc = np.real(amp_acc * np.cos(Omega * t))

    # FFT
    f_ref, fft_ref = FFT_signal(u_exc, dt)

    # PSD
    f_psd, PSD, RMS = compute_PSD_RMS(u_acc, fs)

    idx_max = np.argmax(PSD)
    val_max = PSD[idx_max]
    freq_max = f_psd[idx_max]
    print(f"Fréquence PSD: {freq_max:.4f} Hz")
    print(f"Amplitude PSD: {val_max:.4e} (m/s²)²/Hz")

    # RMS théorique = A_acc / sqrt(2)
    acc_amp_theorique = (Omega ** 2) * abs(amp_acc)
    rms_theorique = acc_amp_theorique / np.sqrt(2)

    print(f"RMS calculé (via PSD)  : {RMS:.4e} m/s²")
    print(f"RMS théorique (approx) : {rms_theorique:.4e} m/s²")

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

    #Plot convergence
    modes = np.arange(1, n_modes + 1)
    plt.figure(figsize=(8,5))
    plt.plot(modes, amp_disp, '-', label='Modal displacement')
    plt.plot(modes, amp_acc, '--', label='Modal acceleration')
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
    plt.plot(modes, err_disp, '-', label='Rel. error displacement [%]')
    plt.plot(modes, err_acc, '--', label='Rel. error acceleration [%]')
    plt.yscale('log')
    plt.xlabel('Number of modes')
    plt.ylabel('Relative error (%)')
    plt.title('Relative error vs number of modes')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    # plot FFT
    plt.figure(figsize=(8, 5))
    plt.stem([f_exc], [np.max(fft_ref)], basefmt=" ")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('FFT at excitation node (single-frequency excitation)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot PSD
    plt.figure(figsize=(8, 5))
    plt.plot(f_psd, PSD, 'b-')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [(m/s²)²/Hz]')
    plt.title('PSD of acceleration (single-frequency harmonic response)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
