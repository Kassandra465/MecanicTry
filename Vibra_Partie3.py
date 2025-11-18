import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from scipy.fft import fft, fftfreq
from Vibra_Partie1 import M_ass, K_ass, frequencies, eigvecs, dofList, dof_map

# FORCE D'IMPACT
F_imp = 7000.0  #N
T_imp = 0.01    #s
node_brown = 2
dof_y = dofList[node_brown][1]
idx_brown = dof_map[dof_y]

#PARAMETRES
gamma_nm = 0.5       #Newmark
beta_nm = 0.25       #Newmark
n_modes = 6          # Nombre de modes considérés
zeta_target = 0.01   # 1% damping

# FRÉQUENCES ET MODES PROPRES
freqs = np.array(frequencies)
omega_n = 2 * np.pi * freqs

#TIME
fs = 2000              # [Hz] Fréquence d’échantillonnage
t_end = 5.0           # [s] Durée d’observation
dt = 1 / fs
t = np.arange(0, t_end, dt)
nt = len(t)

n_dof = M_ass.shape[0]

# ==========================
# Rayleigh damping
# ==========================
def rayleigh_damping(M, K, omega, zeta=zeta_target):
    w1, w2 = omega[0], omega[1]
    beta = (2 * zeta) / (w1 + w2)
    alpha = beta * w1 * w2
    C = alpha * M + beta * K
    zetas = [0.5 * (alpha / w + beta * w) for w in omega[:n_modes]]
    return C, alpha, beta, np.array(zetas)

# ==============
# IMPACT LOAD
# ==============
def build_force_matrix(F_imp, T_imp, idx_brown, n_dof, t_vec):
    n_steps = len(t_vec)
    F_time = np.zeros((n_dof, n_steps))

    for i in range(n_steps):
        if t_vec[i] <= T_imp:
            F_time[idx_brown, i] = F_imp

    return F_time


# =====================
# NEWMARK
# =====================
def Newmark(M, C, K, p, dt, gamma=0.5, beta=0.25):
    n_dof, n_steps = M.shape[0], p.shape[1]
    q = np.zeros((n_steps, n_dof))
    q_dot = np.zeros((n_steps, n_dof))
    q_ddot = np.zeros((n_steps, n_dof))

    # Initial acceleration
    q_ddot[0] = np.linalg.inv(M) @ (p[:, 0] - C @ q_dot[0] - K @ q[0])

    # Effective stiffness
    S_eff = M + gamma * dt * C + (beta * dt**2) * K
    S_eff_inv = np.linalg.inv(S_eff)


    for i in range(1, n_steps):
        # Prediction step
        q_star_dot = q_dot[i - 1] + (1 - gamma) * dt * q_ddot[i - 1]
        q_star = q[i - 1] + dt * q_dot[i - 1] + (0.5 - beta) * dt**2 * q_ddot[i - 1]

        # Compute acceleration
        rhs = p[:, i] - C @ q_star_dot - K @ q_star
        q_ddot[i] = S_eff_inv @ rhs

        # Correction step
        q_dot[i] = q_star_dot + gamma * dt * q_ddot[i]
        q[i] = q_star + beta * dt**2 * q_ddot[i]

    return q, q_dot, q_ddot

# ===============
# FFT FUNCTION
# ===============
def compute_fft(y, dt):
    from scipy.fft import fft, fftfreq
    N = len(y)
    Y = fft(y)
    f = fftfreq(N, dt)[:N//2]
    mag = 2.0 / N * np.abs(Y[:N//2])
    return f, mag

# ===========
# MAIN
# ===========
def main3():
    print(f"Index DOF (dof_y): {dof_y}")
    # Damping
    C, alpha, beta1, zetas = rayleigh_damping(M_ass, K_ass, omega_n, zeta=0.01)
    print(f"\nRayleigh damping: α={alpha:.3e}, β={beta1:.3e}")
    for i, z in enumerate(zetas, start=1):
        print(f"Mode {i}: f = {freqs[i-1]:.3f} Hz, ζ = {z:.4f}")

    # Force

    print(f"Force appliquée sur Noeud {node_brown} (Y). Index Node Brown: {idx_brown}")

    p = build_force_matrix(F_imp, T_imp, idx_brown, n_dof, t)

    # Newmark integration
    q, q_dot, q_ddot = Newmark(M_ass, C, K_ass, p, dt, gamma=0.5, beta=0.25)
    q_exc = q[:, idx_brown]

    print(f"Max q_exc : {np.max(np.abs(q_exc)):.3e} m")

    # FFT of response
    f_exc, fft_exc = compute_fft(q_exc, dt)

    # Plot Displacement
    plt.figure(figsize=(10, 5))
    plt.plot(t, q_exc * 1000, 'b', lw=1.5)  # x1000 pour mm
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [mm]')  # Label mis à jour
    plt.title(f'Transient Response at Node {node_brown} (Y) - Impact {F_imp}N')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # FFT
    plt.figure(figsize=(10, 5))
    plt.semilogy(f_exc, fft_exc * 1000, 'r', lw=1.5)
    plt.xlim(0, 20)
    plt.ylim(0.01, 10)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [mm] (Log Scale)')  # Label mis à jour en mm
    plt.title('FFT of Displacement (Log Scale, Low Freq)')
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Summary Stats
    max_disp_mm = np.max(np.abs(q_exc)) * 1000
    print("\n----- SUMMARY -----")
    print(f"Max Displacement: {max_disp_mm:.2f} mm")
    print(f"Simulation Time: {t_end} s")

    # Summary
    print("\n----- SUMMARY -----")
    print(f"Impact: F = {F_imp:.1f} N, duration = {T_imp*1000:.1f} ms")
    print(f"Time step: {dt:.1e} s, total time: {t_end:.2f} s")
    print(f"Rayleigh coefficients: α = {alpha:.3e}, β = {beta1:.3e}")


if __name__ == "__main__":
    main3()