import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.fftpack import fft

# Constants for terminal colors
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"

# Constants
gravity = 9.81  # m/s²
supporter_weight = 80  # kg
momentum_transfer_ratio = 0.8
damping_ratio_target = 0.005  # 0.5%
dt = 0.01  # Small time step for accurate simulation

# Placeholder mass (M) and stiffness (K) matrices of the system
M = np.eye(6)  # Replace with the actual mass matrix
K = np.eye(6)  # Replace with the actual stiffness matrix

# Calculate natural frequencies and mode shapes
def compute_natural_frequencies_and_modes(M, K):
    eigenvalues, mode_shapes = eigh(K, M)
    natural_frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)
    return natural_frequencies, mode_shapes

# Calculate Rayleigh damping coefficients
def calculate_damping_matrix(M, K, target_damping_ratio, natural_frequencies):
    if len(natural_frequencies) < 2:
        print(f"{RED}ERROR{RESET}: Not enough natural frequencies to calculate damping.")
        return None
    w1, w2 = natural_frequencies[:2] * 2 * np.pi
    beta = 2 * target_damping_ratio / (w1 + w2)
    alpha = beta * w1 * w2
    C = alpha * M + beta * K
    return C

# Calculate damping matrix
natural_frequencies, mode_shapes = compute_natural_frequencies_and_modes(M, K)
C = calculate_damping_matrix(M, K, damping_ratio_target, natural_frequencies)

# Function to calculate impact force
def calculate_impact_force(height, num_people):
    m_total = num_people * supporter_weight
    v_impact = np.sqrt(2 * gravity * height)
    p_impact = momentum_transfer_ratio * m_total * v_impact
    F_amplitude = p_impact / 0.1
    return F_amplitude

# Function to calculate force over time
def force_function(time, F_amplitude):
    return F_amplitude * np.sin(2 * np.pi * 2 * time)

# Mode displacement method
def mode_displacement_method(F_amplitude, observation_nodes, duration=2.0, dt=0.01):
    num_modes = len(natural_frequencies)
    time_steps = int(duration / dt)
    time = np.linspace(0, duration, time_steps)
    response = np.zeros((len(observation_nodes), time_steps))

    for i in range(num_modes):
        omega_i = 2 * np.pi * natural_frequencies[i]
        zeta_i = damping_ratio_target
        F_i = F_amplitude / M[i, i]

        for j, t in enumerate(time):
            q_i = F_i / (omega_i**2) * (1 - np.cos(omega_i * t))
            response[:, j] += q_i * mode_shapes[observation_nodes, i]
    return time, response

# Plot time response
def plot_response(time, response, title="Time Response at Observation Nodes"):
    plt.figure()
    for i, node_response in enumerate(response):
        if node_response.shape[0] != len(time):
            print(f"{RED}ERROR{RESET}: node_response shape {node_response.shape} does not match time shape {len(time)}")
        else:
            plt.plot(time, node_response, label=f"Observation Node {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Calculate and plot response using mode displacement
F_amplitude_1 = calculate_impact_force(0.2, 9)
observation_nodes = [1, 3]  # Example observation nodes
time, response = mode_displacement_method(F_amplitude_1, observation_nodes)
print(f"{RED}Response shape {RESET}: {response.shape}")
plot_response(time, response, "Mode Displacement Method Response - Load Case 1")

# Plot FFT of response
def plot_fft(response, dt, title="Frequency Domain Response"):
    for i, node_response in enumerate(response):
        if node_response.shape[0] != len(time):
            print(f"{RED}ERROR{RESET}: node_response shape {node_response.shape} does not match time shape {len(time)}")
            continue
        N = len(node_response)
        yf = fft(node_response)
        xf = np.fft.fftfreq(N, dt)[:N // 2]
        plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]), label=f"Observation Node {i+1}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Perform FFT and plot
plot_fft(response, dt, "Frequency Domain Response - Load Case 1")

# Newmark integration method
def newmark_integration(M, C, K, F, dt, duration):
    gamma, beta = 1/2, 1/4
    time_steps = int(duration / dt)
    time = np.linspace(0, duration, time_steps)
    dof = len(M)

    u, v, a = np.zeros((dof, time_steps)), np.zeros((dof, time_steps)), np.zeros((dof, time_steps))
    K_eff = M / beta / dt**2 + C * gamma / beta / dt + K

    for n in range(1, time_steps):
        F_n = F(time[n])
        F_eff = F_n + M @ ((u[:,n-1] / beta / dt**2) + v[:,n-1] / beta / dt + a[:,n-1] * (1 / 2 / beta - 1))
        u[:,n] = np.linalg.solve(K_eff, F_eff)
        v[:,n] = gamma / beta / dt * (u[:,n] - u[:,n-1]) + (1 - gamma / beta) * v[:,n-1]
        a[:,n] = 1 / beta / dt**2 * (u[:,n] - u[:,n-1]) - v[:,n-1] / beta / dt - a[:,n-1] * (1 / 2 / beta - 1)

    return time, u

# Newmark integration and response plots
time, u_newmark = newmark_integration(M, C, K, lambda t: force_function(t, F_amplitude_1), dt, duration=2.0)
plot_response(time, u_newmark[observation_nodes], "Newmark Integration Response - Load Case 1")
plot_fft(u_newmark[observation_nodes], dt, "Newmark Frequency Response - Load Case 1")
