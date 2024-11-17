import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Material and cross-sectional properties
E = 210e9  # Young's modulus in Pa
rho = 7800  # Density in kg/m^3
nu = 0.3    # Poisson's ratio
D_outer = 0.15  # Outer diameter in meters
t = 0.005  # Wall thickness in meters
D_inner = D_outer - (2 * t)
A = np.pi * (D_outer**2 - D_inner**2) / 4  # Cross-sectional area
I = np.pi * (D_outer**4 - D_inner**4) / 64  # Moment of inertia
G = E / (2 * (1 + nu)) #Shear Modulus
J = (np.pi / 32) * (D_outer**4 - D_inner**4)

# Geometry of the truss model
heights = [5, 3, 1]  # Heights of the main supports in meters
spacing_x = 4  # Spacing in the x-direction between supports in meters
width_y = 2  # Width of structure in y-direction in meters
num_supporters = 51
weight_per_supporter = 80  # kg


# Total mass of supporters distributed over the end nodes
mass_supporters = num_supporters * weight_per_supporter
mass_per_node = mass_supporters / 18  # Uniformly distributed

def element_matrices(L, A, I, E, rho):
    """Refined stiffness and mass matrices for a 3D beam element."""
    k = np.zeros((12, 12))
    m = np.zeros((12, 12))
    # Implement stiffness and mass matrices calculation for a 3D beam element

    # Rigidité axiale
    k_axial = E * A / L

    # Flexion en deux directions perpendiculaires
    k_flexion = 12 * E * I / L ** 3
    k_rotational_flexion = 6 * E * I / L ** 2
    k_rotational = 4 * E * I / L
    k_cross = 2 * E * I / L

    # Initialisation des matrices de rigidité et de masse
    k = np.zeros((12, 12))
    m = np.zeros((12, 12))

    # Rigidité longitudinale (axiale)
    k[0, 0] = k[6, 6] = k_axial
    k[0, 6] = k[6, 0] = -k_axial

    # Rigidité en flexion
    k[1, 1] = k[7, 7] = k_flexion
    k[5, 5] = k[11, 11] = k_rotational
    k[1, 7] = k[7, 1] = -k_flexion
    k[5, 11] = k[11, 5] = -k_rotational
    k[1, 5] = k[5, 1] = k_rotational_flexion
    k[1, 11] = k[11, 1] = -k_rotational_flexion
    k[5, 7] = k[7, 5] = -k_rotational_flexion
    k[7, 11] = k[11, 7] = k_rotational_flexion

    # Flexion dans l'autre direction
    k[2, 2] = k[8, 8] = k_flexion
    k[4, 4] = k[10, 10] = k_rotational
    k[2, 8] = k[8, 2] = -k_flexion
    k[4, 10] = k[10, 4] = -k_rotational
    k[2, 4] = k[4, 2] = -k_rotational_flexion
    k[2, 10] = k[10, 2] = k_rotational_flexion
    k[4, 8] = k[8, 4] = k_rotational_flexion
    k[8, 10] = k[10, 8] = -k_rotational_flexion

    # Termes de couplage
    k[3, 3] = k[9, 9] = G * J / L
    k[3, 9] = k[9, 3] = -G * J / L

    # Matrice de masse pour la poutre (matrice de consistance de masse pour un élément en 3D)
    m[0, 0] = m[6, 6] = rho * A * L / 3
    m[0, 6] = m[6, 0] = rho * A * L / 6
    m[1, 1] = m[7, 7] = 13 * rho * A * L / 35
    m[1, 7] = m[7, 1] = 9 * rho * A * L / 70
    m[2, 2] = m[8, 8] = 13 * rho * A * L / 35
    m[2, 8] = m[8, 2] = 9 * rho * A * L / 70
    m[4, 4] = m[10, 10] = rho * I * L / 3
    m[5, 5] = m[11, 11] = rho * I * L / 3
    return k, m

k, m = element_matrices(spacing_x, A, I, E, rho)
print("Matrice de rigidité d'un élément:\n", k)
print("Matrice de masse d'un élément:\n", m)

def assemble_global_matrices(num_elements):
    """Assemble the global stiffness and mass matrices."""
    # Tips DOF stands for Degrees of Freedom

    num_nodes = num_elements + 1  # Example number of nodes
    dof_per_node = 6  # Degrees of freedom per node in 3D
    total_dofs = num_nodes * dof_per_node

    K_global = np.zeros((total_dofs, total_dofs)) # Stiffness matrix
    M_global = np.zeros((total_dofs, total_dofs)) # Mass matrix

    for element in range(num_elements):
        L = spacing_x  # Length of elements for simplification
        k_local, m_local = element_matrices(L, A, I, E, rho)

        # Adjusted index range to cover 12 DOFs per element (6 DOFs per node for 2 nodes)
        start_dof = element * dof_per_node
        end_dof = (element + 2) * dof_per_node
        index_range = slice(start_dof, end_dof)

        # Assemble local matrices into the global matrices
        K_global[index_range, index_range] += k_local
        M_global[index_range, index_range] += m_local

    return K_global, M_global

K_global, M_global = assemble_global_matrices(10)
print("Matrice de rigidité globale:\n", K_global)
print("Matrice de masse globale:\n", M_global)

def apply_boundary_conditions(K, M):
    """Apply boundary conditions for clamped nodes."""
    constrained_dofs = [0, 1, 2, 3, 4, 5]  # DOFs for clamped supports

    for dof in constrained_dofs:
        # Only apply to stiffness matrix K
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1  # Keeps K invertible by setting a small value for stability

    # Optionally, handle M matrix separately if needed
    for dof in constrained_dofs:
        M[dof, :] = 0
        M[:, dof] = 0
        M[dof, dof] = 1  # Small value to avoid singularity in M

    return K, M

def compute_natural_frequencies(num_elements):
    """Compute natural frequencies and mode shapes of the structure."""
    K_global, M_global = assemble_global_matrices(num_elements)
    K_global, M_global = apply_boundary_conditions(K_global, M_global)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(K_global, M_global)

    # Natural frequencies (Hz)
    natural_frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

    # Extract the first six natural frequencies and their modes
    first_six_frequencies = natural_frequencies[:6]
    first_six_modes = eigenvectors[:, :6]

    return first_six_frequencies, first_six_modes

frequencies, modes = compute_natural_frequencies(10)
print("Premières six fréquences naturelles (Hz):\n", frequencies)
print("Premières six formes modales:\n", modes)

def plot_mode_shapes(modes, num_nodes, dof_per_node=6):
    """Graphically plot the first six mode shapes."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(6):
        ax = axs[i // 3, i % 3]

        # Extract the mode shape for this frequency
        mode = modes[:, i]
        # Node positions for the plot
        x_positions = np.linspace(0, num_nodes - 1, num_nodes)
        displacement = mode[0::dof_per_node]  # x-displacement for simplicity

        ax.plot(x_positions, displacement, marker='o', linestyle='-', color='b')
        ax.set_title(f"Mode {i+1}")
        ax.set_xlabel("Node Position")
        ax.set_ylabel("Displacement")

    plt.tight_layout()
    plt.show()

plot_mode_shapes(modes, 11)

def convergence_study(min_elements=1, max_elements=20):
    """Perform a convergence study on natural frequencies as a function of element count."""
    first_frequency_values = []  # Store the first natural frequency for each element count
    element_counts = range(min_elements, max_elements + 1)  # Range of element counts to test

    for num_elements in element_counts:
        first_six_frequencies, first_six_modes  = compute_natural_frequencies(num_elements)  # Compute natural frequencies
        first_frequency_values.append(first_six_frequencies[0])  # Record the first natural frequency

    # Plot the convergence of the first natural frequency
    plt.figure(figsize=(10, 6))
    plt.plot(element_counts, first_frequency_values, marker='o', linestyle='-')
    plt.xlabel("Number of Elements per Beam")
    plt.ylabel("First Natural Frequency (Hz)")
    plt.title("Convergence of the First Natural Frequency")
    plt.grid(True)
    plt.show()

convergence_study(1, 15)

num_supporters = 51
weight_per_supporter = 80  # kg
mass_supporters = num_supporters * weight_per_supporter  # Total mass of supporters

def compute_element_mass(rho, A, L):
    """Calculate mass of a single beam element."""
    return rho * A * L

def compute_stadium_mass(num_elements, heights, spacing_x):
    """Compute the total mass of the stadium structure."""
    total_mass_structure = 0

    # Iterate over each main support with its specified height
    for height in heights:
        num_beams = num_elements  # Elements per beam section
        length = height / num_beams  # Length of each element
        element_mass = compute_element_mass(rho, A, length)
        total_mass_structure += element_mass * num_beams

    # Add the horizontal transverse bars between the supports
    for i in range(len(heights) - 1):
        num_beams = num_elements  # Elements per transverse bar
        length = spacing_x / num_beams
        element_mass = compute_element_mass(rho, A, length)
        total_mass_structure += element_mass * num_beams * 2  # For each transverse bar pair

    return total_mass_structure

structure_mass = compute_stadium_mass(57, heights, spacing_x)
print("Masse totale de la structure (kg):", structure_mass)

def compute_total_mass(num_elements, heights, spacing_x, mass_supporters):
    """Calculate total mass of the stadium and supporters using the mass matrix."""
    total_mass_structure = compute_stadium_mass(num_elements, heights, spacing_x)

    # Add the supporters' mass to the structure's total mass
    total_mass = total_mass_structure + mass_supporters
    return total_mass

mass= compute_stadium_mass(57, heights, 0.5)
print(mass)
mass2=compute_total_mass(57, heights, 0.5, mass_supporters)
print(mass2)