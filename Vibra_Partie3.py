import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from scipy.fft import fft, fftfreq
from Vibra_Partie1 import M_ass, K_ass, frequencies, eigvecs, dofList

# PARAMETERS
F_imp = 7000.0
T_imp = 0.01
node_brown_index = 2
dof_y_brown = int(dofList[node_brown_index][1])
idx_brown = dof_y_brown - 1
n_modes = 6          # Nombre de modes considérés

# FRÉQUENCES ET MODES PROPRES
freqs = np.array(frequencies)
omega_n = 2 * np.pi * freqs
Phi = np.array(eigvecs)
n_modes = min(Phi.shape[1], n_modes)

# NEWMARK PARAMETERS
gamma_nm = 0.5
beta_nm = 0.25

dt = 5e-4   #default
t_end = 1.0              # s, durée totale de simulation
t = np.arange(0.0, t_end + dt/2, dt)
nt = len(t)

ndof = M_ass.shape[0]

# ==========================
# Rayleigh damping
# ==========================
zeta_target = 0.01  # 1%
w1 = omega_n[0]
w2 = omega_n[1]

def rayleigh_damping(M, K, omega, zeta=zeta_target):
    w1, w2 = omega[0], omega[1]
    beta = (2 * zeta) / (w1 + w2)
    alpha = beta * w1 * w2
    C = alpha * M + beta * K
    zetas = [0.5 * (alpha / w + beta * w) for w in omega[:n_modes]]
    return C, alpha, beta, np.array(zetas)

# =====================================================
# FORCE VECTOR TIME HISTORY (IMPACT LOAD)
# =====================================================

def build_force_time(F_imp, T_imp, idx, nt, ndof, t):
    F_time = np.zeros((nt, ndof))
    for i, ti in enumerate(t):
        if ti <= T_imp:
            F_time[i, idx] = F_imp
    return F_time

# =====================
# NEWMARK
# =====================

def Newmark(M, C, K, p, h, gamma=0.5, beta=0.25):
    n_dof, n_steps = M.shape[0], p.shape[1]
    q = np.zeros((n_steps, n_dof))
    q_dot = np.zeros((n_steps, n_dof))
    q_ddot = np.zeros((n_steps, n_dof))

    # Initial acceleration
    q_ddot[0] = np.linalg.inv(M) @ (p[:, 0] - C @ q_dot[0] - K @ q[0])

    # Effective stiffness
    S_eff = M + gamma * h * C + (beta * h**2) * K
    S_eff_inv = np.linalg.inv(S_eff)

    # --- Time loop ---
    for i in range(1, n_steps):
        # Prediction step
        q_star_dot = q_dot[i - 1] + (1 - gamma) * h * q_ddot[i - 1]
        q_star = q[i - 1] + h * q_dot[i - 1] + (0.5 - beta) * h**2 * q_ddot[i - 1]

        # Compute acceleration
        rhs = p[:, i] - C @ q_star_dot - K @ q_star
        q_ddot[i] = S_eff_inv @ rhs

        # Correction step
        q_dot[i] = q_star_dot + gamma * h * q_ddot[i]
        q[i] = q_star + beta * h**2 * q_ddot[i]

    return q, q_dot, q_ddot

# =====================================================
# FFT FUNCTION
# =====================================================

def compute_fft(y, dt):
    from scipy.fft import fft, fftfreq
    N = len(y)
    Y = fft(y)
    f = fftfreq(N, dt)[:N//2]
    Y_mag = 2.0 / N * np.abs(Y[:N//2])
    return f, Y_mag

# =====================================================
# MAIN SIMULATION FUNCTION
# =====================================================

def main3():
    # Damping
    C_ass, alphaR, betaR, zetas = rayleigh_damping(M_ass, K_ass, omega_n, zeta=0.01)
    print(f"\nRayleigh damping: α={alphaR:.3e}, β={betaR:.3e}")
    for i, z in enumerate(zetas, start=1):
        print(f"Mode {i}: f = {freqs[i-1]:.3f} Hz, ζ = {z:.4f}")

    # Time definition
    h = 5e-4
    t_end = 1.0
    time = np.arange(0, t_end + h / 2, h)
    n_steps = len(time)
    n_dof = M_ass.shape[0]

    # Force
    p = build_force_time(F_imp, T_imp, idx_brown, n_dof, n_steps, time)

    # Newmark integration
    q, q_dot, q_ddot = Newmark(M_ass, C_ass, K_ass, p, h, gamma=0.5, beta=0.25)
    q_exc = q[:, idx_brown]

    # FFT of response
    f_exc, fft_exc = compute_fft(q_exc, h)

    # Plots
    plt.figure(figsize=(10,5))
    plt.plot(time, q_exc, lw=1.8)
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title('Transient response at excitation node (Impact with damping)')
    plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(f_exc, fft_exc)
    plt.xlim(0, 100)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('FFT of displacement response at excitation node')
    plt.grid(True); plt.tight_layout(); plt.show()

    # Summary
    print("\n----- SUMMARY -----")
    print(f"Impact: F = {F_imp:.1f} N, duration = {T_imp*1000:.1f} ms")
    print(f"Time step: {h:.1e} s, total time: {t_end:.2f} s")
    print(f"Rayleigh coefficients: α = {alphaR:.3e}, β = {betaR:.3e}")


if __name__ == "__main__":
    main3()