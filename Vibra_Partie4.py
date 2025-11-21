import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, solve, lu_factor, lu_solve

from Vibra_Partie1 import M_ass, K_ass, frequencies, eigvecs, dofList, dof_map
from Vibra_Partie3 import C

# PARAMÈTRES
F_imp = 7000.0  # [N]
T_imp = 0.01  # [s]
node_exc = 2
dof_exc_global = dofList[node_exc][1]  # Index Global Y
dof_exc_red = dof_map[dof_exc_global]  # Index Réduit

# TIME
fs = 200              # [Hz] Fréquence d’échantillonnage
t_end = 5.0           # [s] Durée d’observation
dt = 1 / fs
t = np.arange(0, t_end, dt)
nt = len(t)

n_dof = M_ass.shape[0]
node_indices = [2 , 6 , 9]

# ---------------------------
# Translation degrees
# ---------------------------
def get_retained_dofs(node_indices, dof_map, dofList):
    retained_dofs = []

    for node_idx in node_indices:
        global_dofs_node = dofList[node_idx][:3]
        for g_dof in global_dofs_node:
            r_dof = dof_map[g_dof]

            if r_dof != -1:
                retained_dofs.append(r_dof)

    return np.sort(np.unique(retained_dofs))

def build_force_matrix(F_imp, T_imp, node_exc, n_dof, t_vec):
    n_steps = len(t_vec)
    F_time = np.zeros((n_dof, n_steps))

    for i in range(n_steps):
        if t_vec[i] <= T_imp:
            F_time[node_exc, i] = F_imp

    return F_time

# ---------------------------
# Partition Matrix
# ---------------------------
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
def extract_modes(K, M, n_modes):
    M_safe = M + np.eye(M.shape[0]) * 1e-12
    # Résolution
    eigvals, eigvecs = eigh(K, M_safe)
    eigvals = np.real(eigvals)
    eigvals[eigvals < 0] = 0.0
    # Tri
    idx = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Conversion en fréquences (Hz)
    frequencies = np.sqrt(eigvals) / (2 * np.pi)

    # Normalisation modale
    for i in range(eigvecs.shape[1]):
        mode = eigvecs[:, i]
        norm = np.sqrt(mode.T @ M @ mode)
        if norm > 0:
            eigvecs[:, i] = mode / norm

    return frequencies[:n_modes], eigvecs[:, :n_modes]

# ---------------------------
# Guyan-Irons reduction
# ---------------------------
def guyan_irons(M, K, retained_dofs):
    M_RR, M_RC, M_CR, M_CC, R, C = partition_matrix(M, retained_dofs)
    K_RR, K_RC, K_CR, K_CC, _, _ = partition_matrix(K, retained_dofs)
    # compute R_part = -K_CC^{-1} K_CR
    R1 = -np.linalg.inv(K_CC) @ K_CR   # solves K_CC X = K_CR
    # Build R (stacked [I; R_part])
    I = np.eye(len(R))
    R2 = np.vstack([I, R1])
    # Reduced matrices: M_r = R^T * M * R, K_r = R^T * K * R
    M_part = np.block([[M_RR, M_RC], [M_CR, M_CC]])
    K_part = np.block([[K_RR, K_RC], [K_CR, K_CC]])
    M_reduced = R2.T @ M_part @ R2
    K_reduced = R2.T @ K_part @ R2
    return M_reduced, K_reduced, R2, R, C

# ---------------------------
# Craig-Bampton reduction
# ---------------------------
def craig_bampton(M, K, retained_dofs, k_opt):
    M_RR, M_RC, M_CR, M_CC, R, C = partition_matrix(M, retained_dofs)
    K_RR, K_RC, K_CR, K_CC, _, _ = partition_matrix(K, retained_dofs)

    # Modes internes (clamped)
    freqs_I, X_I = extract_modes(K_CC, M_CC, n_modes=k_opt)

    # Partie Statique
    Psi_C = -np.linalg.solve(K_CC, K_CR)

    # Construction matrice T
    n_full = M.shape[0]
    T = np.zeros((n_full, len(R) + k_opt))

    # DOFs retenus = identité
    T[np.ix_(R, range(len(R)))] = np.eye(len(R))
    # Partie statique
    T[np.ix_(C, range(len(R)))] = Psi_C
    # Modes dynamiques internes
    T[np.ix_(C, range(len(R), len(R) + k_opt))] = X_I

    # Projection
    M_red = T.T @ M @ T
    K_red = T.T @ K @ T

    return M_red, K_red, T

# ----------
# Newmark
# ----------
def Newmark(M, C, K, p_time, dt, gamma=0.5, beta=0.25):
    n_dof = M.shape[0]
    nt = p_time.shape[1]

    q = np.zeros((nt, n_dof))      # Déplacement
    qdot = np.zeros((nt, n_dof))   # Vitesse
    qddot = np.zeros((nt, n_dof))  # Accélération

    F0 = p_time[:, 0]
    qddot[0, :] = np.linalg.solve(M, F0)

    M_eff = M + (gamma * dt) * C + (beta * dt**2) * K

    for i in range(0, nt - 1):
        q_pred = q[i, :] + dt * qdot[i, :] + (0.5 - beta) * dt**2 * qddot[i, :]
        qdot_pred = qdot[i, :] + (1 - gamma) * dt * qddot[i, :]

        F_ext = p_time[:, i + 1]
        F_res = F_ext - C @ qdot_pred - K @ q_pred

        a_new = np.linalg.inv(M_eff) @ F_res
        qddot[i + 1, :] = a_new

        q[i + 1, :] = q_pred + beta * dt**2 * a_new
        qdot[i + 1, :] = qdot_pred + gamma * dt * a_new

    return q, qdot, qddot

# ---------------------------
# Find k for Craig-Bampton to reach <1% error on first 6 freqs
# ---------------------------
def find_k(M_full, K_full, retained_dofs, frequencies, k_max=15) :
    n_ref = min(6, len(frequencies))
    if n_ref < 1:
        raise ValueError("freq_ref must contain at least one reference frequency")

    last_freqs = []
    last_errs = []

    for k in range(1, k_max + 1):
        M_red, K_red, T = craig_bampton(M_full, K_full, retained_dofs, k)
        freqs_red, _ = extract_modes(K_red, M_red, n_modes=6)

        if len(freqs_red) < n_ref:
            last_freqs = freqs_red
            last_errs = np.ones(n_ref) * np.inf
            print(f" k={k:2d} -> reduced model returned only {len(freqs_red)} freqs, skipping")
            continue

        # compute relative errors mode-by-mode for the first n_ref modes
        rel_errs = np.abs((freqs_red[:n_ref] - frequencies[:n_ref]) / frequencies[:n_ref])

        max_rel = np.max(rel_errs)
        print(f"  -> Test k={k:2d} : max relative error = {max_rel * 100:.4f} %")

        last_freqs = freqs_red
        last_errs = rel_errs

        if max_rel < 0.01:
            return k, freqs_red, rel_errs

    print(f"  /!\\ No k <= {k_max} found with max rel error < {0.01 * 100:.2f}%")
    return None, last_freqs, last_errs




#==========
# Main
#===========
def main4():
    #DOFs retenus (Noeuds Violets)
    retained_dofs = get_retained_dofs(node_indices, dof_map, dofList)
    print(f"Nombre de DOFs retenus (Taille Guyan) : {len(retained_dofs)}")
    dof_exc_global = dofList[node_exc][1]  # Index Global Y
    dof_exc_red = dof_map[dof_exc_global]  # Index Réduit

    P_full = build_force_matrix(F_imp, T_imp, dof_exc_red, n_dof, t)
    freq_full, _ = extract_modes(K_ass, M_ass, 6)

    # Guyan-Irons
    print("\n--- Guyan-Irons ---")
    t0 = time.perf_counter()
    M_guyan, K_guyan, R_guyan, _, _ = guyan_irons(M_ass, K_ass, retained_dofs)
    t_guyan_red = time.perf_counter() - t0

    # Fréquences réduites
    freq_guyan, _ = extract_modes(K_guyan, M_guyan, 6)
    err_guyan = 100 * np.abs(freq_guyan - freq_full) / freq_full

    print(f"Fréquences Guyan : {np.round(freq_guyan, 4)}")
    print(f"Erreur Rel. (%)  : {np.round(err_guyan, 4)}")
    print(f"Temps réduction  : {t_guyan_red:.4f} s")

    # Simulation Guyan
    print("Simu Guyan...")
    # Projection Force & Amortissement
    P_guyan = R_guyan.T @ P_full
    C_guyan = R_guyan.T @ C @ R_guyan

    t0 = time.perf_counter()
    q_guyan_red, _, _ = Newmark(M_guyan, C_guyan, K_guyan, P_guyan, dt)
    t_guyan_sim = time.perf_counter() - t0

    # Reconstruction physique
    u_guyan_phys = (R_guyan @ q_guyan_red.T).T
    u_guyan = u_guyan_phys[:, dof_exc_red]

    # Craig-Bampton
    print("\n--- Craig-Bampton ---")
    t0 = time.perf_counter()
    k_opt, freq_cb, err_cb_list = find_k(M_ass, K_ass, retained_dofs, freq_full, k_max=10)
    t_cb_search = time.perf_counter() - t0

    # On reconstruit les matrices finales pour ce k optimal
    M_cb, K_cb, R_cb = craig_bampton(M_ass, K_ass, retained_dofs, k_opt)

    print(f"k optimal trouvé   : {k_opt}")
    print(f"Fréquences Craig-Bampton (Hz) : {np.round(freq_cb, 4)}")
    print(f"Temps recherche    : {t_cb_search:.4f} s")

    # Simulation Craig-Bampton
    P_cb = R_cb.T @ P_full
    C_cb = R_cb.T @ C @ R_cb

    t0 = time.perf_counter()
    q_cb_red, _, _ = Newmark(M_cb, C_cb, K_cb, P_cb, dt)
    t_cb_sim = time.perf_counter() - t0

    u_cb_phys = (R_cb @ q_cb_red.T).T
    u_cb = u_cb_phys[:, dof_exc_red]

    # Full Model (Référence)
    t0 = time.perf_counter()
    # Attention : Newmark sur le modèle complet est lent
    q_full, _, _ = Newmark(M_ass, C, K_ass, P_full, dt)
    t_full_sim = time.perf_counter() - t0

    u_full = q_full[:, dof_exc_red]

    # Affichage et Comparaisons

    # Conversion en mm
    u_full_mm = u_full * 1000
    u_guyan_mm = u_guyan * 1000
    u_cb_mm = u_cb * 1000

    print("\n----- BILAN PERFORMANCE -----")
    print(f"{'Méthode':<15} | {'Taille (DOFs)':<12} | {'Temps Simu (s)':<15} | {'Speedup':<10}")
    print("-" * 60)
    print(f"{'Full':<15} | {M_ass.shape[0]:<12} | {t_full_sim:.4f}          | 1.0x")
    print(f"{'Guyan':<15} | {M_guyan.shape[0]:<12} | {t_guyan_sim:.4f}          | {t_full_sim / t_guyan_sim:.1f}x")
    print(f"{'Craig-Bampton':<15} | {M_cb.shape[0]:<12} | {t_cb_sim:.4f}          | {t_full_sim / t_cb_sim:.1f}x")

    # Plots
    # Guyan-Irons model
    plt.figure(figsize=(10, 4))
    plt.plot(t, u_guyan_mm, 'b--', lw=1.5)
    plt.title(f"Guyan-Irons Reduced Model Response at Node {node_exc} (Y)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Craig-Bampton model
    plt.figure(figsize=(10, 4))
    plt.plot(t, u_cb_mm, 'r-', lw=2)
    plt.title(f"Craig-Bampton Reduced Model Response at Node {node_exc} (Y), k={k_opt}")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #  Global Plot (log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(t, np.abs(u_full_mm), 'k-', lw=2, alpha=0.7, label='Full (Ref)')
    plt.plot(t, np.abs(u_guyan_mm), 'b--', lw=1.5, label='Guyan')
    plt.plot(t, np.abs(u_cb_mm), 'r:', lw=2, label=f'CB (k={k_opt})')
    plt.yscale("log")
    plt.title(f"Transient Response (Log scale) at Node {node_exc} (Y) - Impact {F_imp}N")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm] (log scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.show()

    # Zoom Impact
    plt.figure(figsize=(10, 6))
    plt.plot(t, u_full_mm, 'k-', lw=2, alpha=0.6, label='Full')
    plt.plot(t, u_guyan_mm, 'b--', lw=1.5, label='Guyan')
    plt.plot(t, u_cb_mm, 'r:', lw=2, label='CB')
    plt.xlim(0, 0.2)
    plt.title("Zoom Impact")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [mm]")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    results = main4()
