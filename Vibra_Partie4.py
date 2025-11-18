import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, solve, lu_factor, lu_solve

from Vibra_Partie1 import M_ass, K_ass, frequencies, eigvecs, dofList

node_indices = [2 , 6 , 9]

# ---------------------------
# Translation degrees
# ---------------------------
def translation_dof_indices(dofList, node_indices):
    dof_indices = []
    for node in node_indices:
        entry = dofList[node - 1]   # earlier code used node-1
        for j in range(3):
            dof_idx = int(entry[j]) - 1
            dof_indices.append(dof_idx)
    return dof_indices

def partition_matrix(matrix, retained_dofs):
    n = matrix.shape[0]
    R = list(retained_dofs)
    C = [i for i in range(n) if i not in R]
    M_RR = matrix[np.ix_(R, R)]
    M_RC = matrix[np.ix_(R, C)]
    M_CR = matrix[np.ix_(C, R)]
    M_CC = matrix[np.ix_(C, C)]
    return M_RR, M_RC, M_CR, M_CC, R, C

# ---------------------------
# Natural frequencies & modes
# ---------------------------
def compute_natural_frequencies_and_modes(M, K, nmodes):
    vals, vecs = eigh(K, M)   # eigenvalues
    vals = np.real(vals)
    positive = vals > 1e-12
    vals = vals[positive]
    vecs = vecs[:, positive]
    omegas = np.sqrt(np.abs(vals))    # rad/s
    freqs = omegas / (2.0 * np.pi)   # Hz
    # sort
    order = np.argsort(freqs)
    freqs = freqs[order]
    vecs = vecs[:, order]
    if nmodes is not None:
        freqs = freqs[:nmodes]
        vecs = vecs[:, :nmodes]
    return freqs, vecs

# ---------------------------
# Guyan-Irons reduction
# ---------------------------
def guyan_irons(M, K, retained_dofs):
    M_RR, M_RC, M_CR, M_CC, R, C = partition_matrix(M, retained_dofs)
    K_RR, K_RC, K_CR, K_CC, _, _ = partition_matrix(K, retained_dofs)
    # compute R_part = -K_CC^{-1} K_CR
    K_CC_inv_K_CR = solve(K_CC, K_CR)   # solves K_CC X = K_CR
    R_part = -K_CC_inv_K_CR
    # Build R (stacked [I; R_part])
    I = np.eye(len(R))
    Rmat = np.vstack([I, R_part])
    # Reduced matrices: M_r = R^T * M * R, K_r = R^T * K * R
    bigM = np.block([[M_RR, M_RC], [M_CR, M_CC]])
    bigK = np.block([[K_RR, K_RC], [K_CR, K_CC]])
    M_reduced = Rmat.T @ bigM @ Rmat
    K_reduced = Rmat.T @ bigK @ Rmat
    return M_reduced, K_reduced, Rmat, R, C

# ---------------------------
# Craig-Bampton reduction
# ---------------------------
def craig_bampton(M, K, retained_indices, n_clamped_modes):
    # Partition
    M_BB, M_BI, M_IB, M_II, Ridx, Cidx = partition_matrix(M, retained_indices)
    K_BB, K_BI, K_IB, K_II, _, _ = partition_matrix(K, retained_indices)
    # Solve clamped modes on internal (I) DOFs: K_II x = lambda M_II x
    freqs_I, X_I = compute_natural_frequencies_and_modes(M_II, K_II, nmodes=n_clamped_modes)
    # Build transformation:
    R_part = -solve(K_II, K_IB)    # -K_II^{-1} K_IB
    I_B = np.eye(len(Ridx))
    Re = np.vstack([I_B, R_part])                 # maps boundary dofs (B) into full internal/full vector
    # Rright maps modal coordinates (clamped modes) into DOFs (zero on B, X_I on I)
    zeros_B_XI = np.zeros((len(Ridx), X_I.shape[1]))
    Rright = np.vstack([zeros_B_XI, X_I])
    # R = [Re, Rright]
    Rmat = np.hstack([Re, Rright])
    # Build big M/K and project
    bigM = np.block([[M_BB, M_BI], [M_IB, M_II]])
    bigK = np.block([[K_BB, K_BI], [K_IB, K_II]])
    M_reduced = Rmat.T @ bigM @ Rmat
    K_reduced = Rmat.T @ bigK @ Rmat
    return M_reduced, K_reduced, Rmat

# ---------------------------
# Newmark integrator
# ---------------------------
def Newmark(M, C, K, p_time, time_vec, gamma=0.5, beta=0.25):
    n_steps = p_time.shape[1]
    ndof = M.shape[0]
    h = time_vec[1] - time_vec[0]

    q = np.zeros((n_steps, ndof))
    qdot = np.zeros((n_steps, ndof))
    qddot = np.zeros((n_steps, ndof))

    # initial acceleration
    qddot[0, :] = solve(M, p_time[:, 0] - C @ qdot[0, :] - K @ q[0, :])

    S_eff = M + gamma * h * C + (beta * h**2) * K
    lu, piv = lu_factor(S_eff)

    for i in range(1, n_steps):
        q_star_dot = qdot[i - 1, :] + (1 - gamma) * h * qddot[i - 1, :]
        q_star = q[i - 1, :] + h * qdot[i - 1, :] + (0.5 - beta) * h**2 * qddot[i - 1, :]

        rhs = p_time[:, i] - C @ q_star_dot - K @ q_star
        qddot[i, :] = lu_solve((lu, piv), rhs)
        qdot[i, :] = q_star_dot + gamma * h * qddot[i, :]
        q[i, :] = q_star + beta * h**2 * qddot[i, :]

    return q, qdot, qddot

# ---------------------------
# Find k for Craig-Bampton to reach <1% error on first 6 freqs
# ---------------------------
def find_min_k_for_1pct(M_full, K_full, retained_indices, freq_ref, max_k=30):

    for k in range(1, max_k + 1):
        M_r, K_r, Rmat = None, None, None
        M_r, K_r, Rmat = craig_bampton(M_full, K_full, retained_indices, n_clamped_modes=k)
        freqs_r, _ = compute_natural_frequencies_and_modes(M_r, K_r, nmodes=6)
        # pad if too few freq returned
        if len(freqs_r) < 6:
            continue
        rel_err = np.abs((freqs_r[:6] - freq_ref[:6]) / freq_ref[:6])
        if np.max(rel_err) < 0.01:
            return k, freqs_r, rel_err
    return None, None, None

# ---------------------------
# Main procedure
# ---------------------------
def main4():
    print("=== reduction and comparison ===")
    retained_dofs = translation_dof_indices(dofList, node_indices)
    print("Retained DOFs (0-based):", retained_dofs)

    # Full model frequencies (first 6)
    freq_full, modes_full = compute_natural_frequencies_and_modes(M_ass, K_ass, nmodes=6)
    print("Full model first 6 frequencies (Hz):", np.round(freq_full, 6))

    # Guyan-Irons reduction
    t0 = time.perf_counter()
    M_guyan, K_guyan, R_guyan, Ridx, Cidx = guyan_irons(M_ass, K_ass, retained_dofs)
    t_guyan = time.perf_counter() - t0
    freq_guyan, modes_guyan = compute_natural_frequencies_and_modes(M_guyan, K_guyan, nmodes=6)
    print("Guyan-Irons 6 freqs (Hz):", np.round(freq_guyan, 6), " (time: {:.3f}s)".format(t_guyan))

    # Craig-Bampton: find minimal k
    t0 = time.perf_counter()
    k_found, freqs_cb, rel_err_cb = find_min_k_for_1pct(M_ass, K_ass, retained_dofs, freq_full, max_k=30)
    t_cb_freqs = time.perf_counter() - t0
    if k_found is None:
        print("Craig-Bampton: could not reach <1% error up to k=30")
        k_found = 6
    else:
        print(f"Craig-Bampton: found k = {k_found} modes (time {t_cb_freqs:.3f}s), rel_err (first 6):", np.round(rel_err_cb, 6))

    # Build Craig-Bampton reduced matrices with the found k
    M_cb, K_cb, R_cb = craig_bampton(M_ass, K_ass, retained_dofs, n_clamped_modes=k_found)
    freq_cb, modes_cb = compute_natural_frequencies_and_modes(M_cb, K_cb, nmodes=6)
    print("Craig-Bampton 6 freqs (Hz):", np.round(freq_cb, 6))

    # Compare frequency lists
    print("\nComparison (Full vs Guyan vs Craig-Bampton):")
    for i in range(6):
        print(f"Mode {i+1:2d}: Full {freq_full[i]:8.4f} Hz | Guyan {freq_guyan[i]:8.4f} Hz | CB {freq_cb[i]:8.4f} Hz")


    exc_dof_full = retained_dofs[0]

    fs = 200  # [Hz] Fréquence d’échantillonnage
    t_end = 5.0  # [s] Durée d’observation
    dt = 1 / fs
    t = np.arange(0.0, t_end + dt / 2, dt)
    nt = len(t)


    p_full = np.zeros((M_ass.shape[0], nt))

    # Damping for full model: Rayleigh using first two modes → 1% target
    # convert full natural freqs to rad/s
    freq_all, _ = compute_natural_frequencies_and_modes(M_ass, K_ass, nmodes=6)
    w1, w2 = 2*np.pi*freq_all[0], 2*np.pi*freq_all[1]
    zeta = 0.01
    beta = (2*zeta) / (w1 + w2)
    alpha = beta * w1 * w2
    C_full = alpha * M_ass + beta * K_ass

    # Newmark full
    t0 = time.perf_counter()
    q_full, qdot_full, qddot_full = Newmark(M_ass, C_full, K_ass, p_full, t)
    t_newmark_full = time.perf_counter() - t0
    print("Newmark on full model time: {:.3f}s".format(t_newmark_full))

    # Newmark on Guyan model
    # Project p_full onto reduced DOFs: p_r = R^T p_full (R for Guyan keeps mapping)
    # For Guyan we used R_guyan mapping from reduced coords to full coords, so projection p_red = R_guyan.T @ p_full(full DOF)
    # We need p_red_time shape = (n_red, n_steps)
    p_guyan = R_guyan.T @ p_full
    # Damping: use Rayleigh on reduced M,K with same alpha, beta
    C_guyan = alpha * M_guyan + beta * K_guyan
    t0 = time.perf_counter()
    q_g, qdot_g, qddot_g = Newmark(M_guyan, C_guyan, K_guyan, p_guyan, t)
    t_newmark_guyan = time.perf_counter() - t0
    print("Newmark on Guyan model time: {:.3f}s".format(t_newmark_guyan))

    # Newmark on Craig-Bampton model
    p_cb = R_cb.T @ p_full
    C_cb = alpha * M_cb + beta * K_cb
    t0 = time.perf_counter()
    q_cb, qdot_cb, qddot_cb = Newmark(M_cb, C_cb, K_cb, p_cb, t)
    t_newmark_cb = time.perf_counter() - t0
    print("Newmark on Craig-Bampton model time: {:.3f}s".format(t_newmark_cb))

    # Reconstruct full-space response at the excitation DOF for comparison:
    # For Guyan: full_disp = R_guyan @ q_g.T (R maps reduced->full), q_g is (n_steps x n_red)
    disp_full_from_guyan = (R_guyan @ q_g.T).T   # shape (n_steps, n_full_dof)
    disp_full_from_cb = (R_cb @ q_cb.T).T

    u_full = q_full[:, exc_dof_full]
    u_guyan_recon = disp_full_from_guyan[:, exc_dof_full]
    u_cb_recon = disp_full_from_cb[:, exc_dof_full]

    # Plots: compare time series
    plt.figure(figsize=(10,5))
    plt.plot(t, u_full, label='Full (reference)', lw=1.6)
    plt.plot(t, u_guyan_recon, '--', label='Guyan-Irons recon', lw=1.2)
    plt.plot(t, u_cb_recon, ':', label=f'Craig-Bampton (k={k_found}) recon', lw=1.2)
    plt.xlabel('Time [s]'); plt.ylabel('Displacement [m]')
    plt.title('Transient response at excitation DOF (full vs reduced)')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # Print summary
    print("\nSummary of timings (s):")
    print("Guyan frequency reduction time: {:.3f}s".format(t_guyan))
    print("Craig-Bampton frequency search time: {:.3f}s".format(t_cb_freqs))
    print("Newmark full: {:.3f}s | Guyan: {:.3f}s | Craig-Bampton: {:.3f}s".format(t_newmark_full, t_newmark_guyan, t_newmark_cb))
    # Return important outputs
    return {
        'freq_full': freq_full,
        'freq_guyan': freq_guyan,
        'freq_cb': freq_cb,
        'k_cb': k_found,
        'u_full': u_full,
        'u_guyan': u_guyan_recon,
        'u_cb': u_cb_recon,
        'time_vec': t
    }

# Run when module executed directly
if __name__ == "__main__":
    results = main4()
