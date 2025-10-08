import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, eigh
from scipy import linalg

# ==========================
# Matériaux et propriétés
# ==========================
rho = 7800.0           # kg/m3
E = 210e9              # Pa
nu = 0.3
G = E / (2*(1+nu))     # Module de cisaillement

div = 0  # Number of beams' divisions (=> (div+1) is the number of elements per beam).

# ==========================
# Géométrie de la structure
# ==========================
main_nodes = np.array([
    [1.5, 0, 0], [4.5, 0, 0], [7.5, 0, 0], [10.5, 0, 0], [13.5, 0, 0],
    [15, 0, 1], [12, 0, 1], [9, 0, 1], [6, 0, 1], [3, 0, 1], [0, 0, 1],
    [1.5, 4, 0], [4.5, 4, 0], [7.5, 4, 0], [10.5, 4, 0], [13.5, 4, 0],
    [15, 4, 1], [12, 4, 1], [9, 4, 1], [6, 4, 1], [3, 4, 1], [0, 4, 1]
])

main_beams = np.array([
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 0],
    [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [11, 21],
    [4, 6], [3, 6], [3, 7], [2, 7],  [2, 8], [1, 8],  [0, 9], [1, 9], [0, 11],
    [1, 12],  [2, 13],  [3, 14],  [4, 15],[15, 17], [6, 17], [14, 17],  [14, 18], [13, 18],
    [7, 18],  [13, 19], [12, 19], [8, 19],  [12, 20], [11, 20],[9, 20]
])

# ==========================
# Section differentes
# ==========================
def section(D, t):
    Di = D - 2 * t
    A = np.pi * (D**2 - Di**2) / 4.0
    Iy = np.pi * (D**4 - Di**4) / 64.0
    Iz = Iy
    Jx = 2 * Iy
    return A, Iy, Iz, Jx


def define_element_sections(main_beams):
    # Cadres rouges
    A_f, Iy_f, Iz_f, Jx_f = section(0.120, 0.005)
    # Poutres bleues
    A_s, Iy_s, Iz_s, Jx_s = section(0.070, 0.003)

    elem_section_dict = {}

    for i in range(len(main_beams)):
        if i < 22:
            elem_section_dict[i] = {
                "type": "frame",
                "A": A_f, "Iy": Iy_f, "Iz": Iz_f, "Jx": Jx_f
            }
        else:
            elem_section_dict[i] = {
                "type": "support",
                "A": A_s, "Iy": Iy_s, "Iz": Iz_s, "Jx": Jx_s
            }

    return elem_section_dict


# ==========================
# Division en elements per beam
# ==========================
# Génération des nœuds et poutres avec divisions
"""
if (div > 0):

    current_node_index = len(main_nodes)  # Index for new nodes

    for element in main_beams:
        node1, node2 = element
        coord1 = main_nodes[node1]
        coord2 = main_nodes[node2]

        # Add intermediate nodes
        previous_node = node1
        for i in range(1, div + 1):
            interpolated_coord = coord1 + i * (coord2 - coord1) / (div + 1)
            new_nodes.append(interpolated_coord)
            current_node_index += 1
            new_beams.append([previous_node, current_node_index])
            previous_node = current_node_index
        # Add the final segment
        new_beams.append([previous_node, node2])
    main_nodes_add = np.vstack((main_nodes, np.array(new_nodes)))
    main_beams_add = np.array(new_beams)
else:
    main_nodes_add = main_nodes
    main_beams_add = main_beams

# Convert to lists if needed
nodeList = main_nodes_add.tolist()
elemList = main_beams_add.tolist()
"""

def subdivide_mesh(main_nodes, main_beams, div, elem_section_type):

    if div == 0:
        return main_nodes.copy(), main_beams.copy(), elem_section_type.copy()

    new_nodes = []
    new_beams = []
    new_section_type = {}
    current_node_index = len(main_nodes)
    new_elem_index = 0

    for i, (node1, node2) in enumerate(main_beams):
        coord1 = main_nodes[node1]
        coord2 = main_nodes[node2]

        # Identify type of current beam (red/blue)
        beam_type = elem_section_type.get(i, 'frame')

        previous_node = node1

        # Create intermediate nodes
        for k in range(1, div + 1):
            new_coord = coord1 + k * (coord2 - coord1) / (div + 1)
            new_nodes.append(new_coord)

            # Create a new subdivided beam
            new_beams.append([previous_node, current_node_index])
            new_section_type[new_elem_index] = beam_type  # same type as parent
            new_elem_index += 1

            previous_node = current_node_index
            current_node_index += 1

        # Connect last new node to the original end
        new_beams.append([previous_node, node2])
        new_section_type[new_elem_index] = beam_type
        new_elem_index += 1

    # Combine original and new nodes
    new_nodes_array = np.vstack((main_nodes, np.array(new_nodes)))
    new_beams_array = np.array(new_beams, dtype=int)

    return new_nodes_array, new_beams_array, new_section_type

"""
def section(D, t):
    Retourne A, Iy, Iz, Jx pour un tube circulaire creux
    Di = D - 2*t
    A = np.pi * (D**2 - Di**2) / 4.0
    Iy = np.pi * (D**4 - Di**4) / 64.0
    Iz = Iy
    Jx = 2*Iy
    return A, Iy, Iz, Jx

nbr_elem = 47 * (div + 1)

for i in range (nbr_elem) :
    if i < (22 * (div +1)) :
    A_f, Iy_f, Iz_f, Jx_f = section(0.120, 0.005) # Cadres rouges
else :
    A_s, Iy_s, Iz_s, Jx_s = section(0.070, 0.003) # Poutres bleues
"""

# ==========================
# DoFs
# ==========================
def create_dof_list(n_nodes: int):
    dof_list = []

    for i in range(n_nodes):
        start = i * 6
        node_dofs = list(range(start, start + 6))
        dof_list.append(node_dofs)

    return dof_list

# ==========================
# Matrice de rotation
# ==========================

def Rotation_fonction(node1, node2):
    x_axis = (node2 - node1)
    L = np.linalg.norm(x_axis)
    ex = x_axis / L
    # Choix d’un vecteur arbitraire non colinéaire
    arbitrary = np.array([0,0,1]) if abs(ex[2])<0.9 else np.array([0,1,0])
    ey = np.cross(arbitrary, ex); ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.vstack((ex,ey,ez))

# ==========================
# Matrices élémentaires
# ==========================
def K_el(E, G, A, l, Iy, Iz, Jx):
    return np.array([
        [ E*A/l, 0, 0, 0,0, 0, -E*A/l, 0, 0, 0, 0, 0],
        [0, 12*E*Iz/l**3, 0, 0, 0, 6*E*Iz/l**2, 0, -12*E*Iz/l**3, 0, 0, 0, 6*E*Iz/l**2],
        [0, 0, 12*E*Iy/l**3, 0, -6*E*Iy/l**2, 0, 0, 0, -12*E*Iy/l**3, 0, -6*E*Iy/l**2, 0],
        [0, 0, 0, G*Jx/l, 0, 0, 0, 0, 0, -G*Jx/l, 0, 0],
        [0, 0, -6*E*Iy/l**2, 0, 4*E*Iy/l, 0, 0,  0, 6*E*Iy/l**2, 0, 2*E*Iy/l, 0],
        [0, 6*E*Iz/l**2, 0, 0, 0, 4*E*Iz/l, 0, -6*E*Iz/l**2, 0, 0,  0, 2*E*Iz/l],
        [-E*A/l, 0, 0, 0, 0, 0, E*A/l, 0, 0, 0, 0, 0],
        [0, -12*E*Iz/l**3, 0, 0, 0, -6*E*Iz/l**2, 0, 12*E*Iz/l**3, 0, 0, 0, -6*E*Iz/l**2],
        [0, 0, -12*E*Iy/l**3, 0, 6*E*Iy/l**2, 0, 0, 0, 12*E*Iy/l**3, 0, 6*E*Iy/l**2, 0],
        [0, 0, 0, -G*Jx/l, 0, 0, 0, 0, 0, G*Jx/l, 0, 0],
        [0, 0, -6*E*Iy/l**2, 0, 2*E*Iy/l, 0, 0, 0, 6*E*Iy/l**2, 0, 4*E*Iy/l, 0],
        [0, 6*E*Iz/l**2, 0, 0, 0, 2*E*Iz/l, 0, -6*E*Iz/l**2, 0, 0, 0, 4*E*Iz/l]
    ], dtype=float)

def M_el(rho, A, l, Iy, Iz):
    I = 0.5*(Iy+Iz)
    r = np.sqrt(I/A) if A>0 else 0.0
    return rho*A*l * np.array([
        [ 1/3, 0, 0, 0, 0, 0, 1/6, 0, 0, 0, 0, 0],
        [ 0, 13/35, 0, 0, 0, 11*l/210, 0, 9/70, 0, 0, 0, -13*l/420],
        [ 0, 0, 13/35, 0, -11*l/210, 0, 0, 0, 9/70, 0, 13*l/420, 0],
        [ 0, 0, 0, r**2/3, 0, 0, 0, 0, 0, r**2/6, 0, 0],
        [ 0, 0, -11*l/210, 0, l**2/105, 0, 0, 0, -13*l/420, 0, -l**2/140, 0],
        [ 0, 11*l/210, 0, 0, 0, l**2/105, 0, 13*l/420, 0, 0, 0, -l**2/140],
        [ 1/6, 0, 0, 0, 0, 0, 1/3, 0, 0, 0, 0, 0],
        [ 0, 9/70, 0, 0, 0, 13*l/420, 0, 13/35, 0, 0, 0, -11*l/210],
        [ 0, 0, 9/70, 0, -13*l/420, 0, 0, 0, 13/35, 0, 11*l/210, 0],
        [ 0, 0, 0, r**2/6, 0, 0, 0, 0, 0, r**2/3, 0, 0],
        [ 0, 0, 13*l/420, 0, -l**2/140, 0, 0, 0, 11*l/210, 0, l**2/105, 0],
        [ 0, -13*l/420, 0, 0, 0, -l**2/140, 0, -11*l/210, 0, 0, 0, l**2/105]
    ], dtype=float)

# ==========================
# Assemblage global
# ==========================
def assemble_matrices(new_nodes_array, beams, elem_section_dict, E, G, rho):
    n_nodes = len(new_nodes_array)
    dof_list = create_dof_list(n_nodes)
    nbr_dof = n_nodes * 6
    K_global = np.zeros((nbr_dof, nbr_dof))
    M_global = np.zeros((nbr_dof, nbr_dof))

    for e, (n1, n2) in enumerate(beams):
        node1 = np.array(new_nodes_array[n1])
        node2 = np.array(new_nodes_array[n2])
        L = np.linalg.norm(node2 - node1)

        # ---- section properties pour cet élément
        props = elem_section_dict[e]
        A = props['A']; Iy = props['Iy']; Iz = props['Iz']; Jx = props['Jx']

        # ---- matrices locales
        Ke_local = K_el(E, G, A, L, Iy, Iz, Jx)
        Me_local = M_el(rho, A, L, Iy, Iz)

        # ---- rotation et transformation
        R = Rotation_fonction(node1, node2)   # gère inclinaison en 3D
        T = block_diag(R, R, R, R)  # 12x12

        Ke_g = T.T @ Ke_local @ T
        Me_g = T.T @ Me_local @ T

        # ---- assemblage dans K_global/M_global
        loc = dof_list[n1] + dof_list[n2]   # 12 indices
        for i_loc in range(12):
            for j_loc in range(12):
                K_global[loc[i_loc], j_loc] += Ke_g[i_loc, j_loc]
                M_global[loc[i_loc], j_loc] += Me_g[i_loc, j_loc]

    return K_global, M_global


 #compter le nombre d'élément à partir des grosses beam


def assemble_global_matrices(K_s, M_s, dofList):
    nodes_with_mass = [5, 6, 7, 8, 16, 17, 18, 19] # masses du toit
    roof_mass = 500.0 / len(nodes_with_mass)
    M_global = M_s
    K_global = K_s

    # --- Ajout des masses concentrées sur les DDL de translation ---
    for nd in nodes_with_mass:
        for dof in dofList[nd][:3]:  # ux, uy, uz uniquement
            M_global[dof, dof] += roof_mass

    # --- Application des encastrements ---
    fixed_nodes = [5, 10, 16, 21]  # encastrements
    fixed_dofs = [d for nd in fixed_nodes for d in dofList[nd]]

    # Suppression des lignes et colonnes correspondantes
    K_ass = np.delete(np.delete(K_global, fixed_dofs, axis=0), fixed_dofs, axis=1)
    M_ass = np.delete(np.delete(M_global, fixed_dofs, axis=0), fixed_dofs, axis=1)

    return M_ass, K_ass

# ==========================
# Extraction fréquences propres
# ==========================
def extract_modes(K_reduced, M_reduced, n_modes=6):
    # --- Sécurité : symétrisation numérique ---
    K_reduced = 0.5 * (K_reduced + K_reduced.T)
    M_reduced = 0.5 * (M_reduced + M_reduced.T)

    # --- Résolution du problème généralisé ---
    eigvals, eigvecs = eigh(K_reduced, M_reduced)

    # --- Nettoyage des valeurs propres négatives (artefacts numériques) ---
    eigvals = np.real(eigvals)
    eigvals[eigvals < 0] = 0.0

    # --- Tri croissant des fréquences ---
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # --- Conversion en fréquences [Hz] ---
    frequencies = np.sqrt(eigvals) / (2 * np.pi)

    # --- Normalisation des modes ---
    for i in range(eigvecs.shape[1]):
        mode = eigvecs[:, i]
        norm = np.sqrt(np.dot(mode.T, M_reduced @ mode))
        eigvecs[:, i] = mode / norm if norm > 0 else mode

    return frequencies[:n_modes], eigvecs[:, :n_modes]


# Fonction pour effectuer l'étude de convergence
def convergence_study():
    num_elements_per_beam = div + 1
    convergence_results = []

    for num_elem in range(num_elements_per_beam):

        # Assemblage des matrices globales
        M_global, K_global = assemble_global_matrices()

        # Extraction des fréquences naturelles et des modes associés
        frequencies, eigenvectors = extract_modes(K_global, M_global)

        # Stockage des résultats
        convergence_results.append((num_elem, frequencies[:6]))

    return convergence_results

def Calcul_Eig(M_global1, K_global1, draw=True):
    # Calcul Eigen values and vectors (Shapes).
    w_carre, eigenModes = linalg.eigh(K_global1, M_global1)
    w_carre = np.array(w_carre)  # np.array is used for doing some operations.
    w = np.sqrt(w_carre)  # Pay attention that this w is in [rad/s].

    eigenFreq = w / (2 * math.pi)  # For getting the normal frequencies in [Hz].

    sorted_indices = np.argsort(eigenFreq)  # Sort eigenFreq and get the indices for sorting.
    sorted_eigenFreq = eigenFreq[sorted_indices]

    sorted_eigenModes = eigenModes[:, sorted_indices]  # Rearrange columns of eigenModes based on sorted indices.

    print(f'\n################### sorted Eigen/Normal Frequencies (div = {div}): #####################')
    print(sorted_eigenFreq[:6].real)

    return M_global1, K_global1, sorted_eigenFreq, sorted_eigenModes


#==================#
#       Main       #
#==================#
freqencies, modes = extract_modes(["K_red"],["M_red"], n_modes=6)
print("Premières fréquences [Hz] :", freqencies)
for i,f in enumerate(freqencies,1):
    print(f"  Mode {i} : {f:.4f} Hz")


# Fonction pour tracer l'étude de convergence
def plot_convergence_study(element_counts, frequencies):
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(element_counts, frequencies[:, i], marker='o', label=f'Mode {i + 1}')
    plt.xlabel('Number of Elements per Beam')
    plt.ylabel('Natural Frequency (Hz)')
    plt.title('Convergence Study')
    plt.legend()
    plt.grid()
    plt.show()

#Draw Points Fct
def plot_mode_shape(nodes, elements, mode_vector, mode_number, frequency, scale_factor=5.0):
    """Plot the mode shape for a given mode number"""
    degress_of_freedom = 6
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get original and deformed node coordinates
    original_coords = nodes  # Skip node ID

    full_mode_vector = np.zeros((len(nodes), degress_of_freedom))
    free_dof_idx = 0
    fixed_nodes = [5, 10, 16, 21]
    fixed_dofs = [degress_of_freedom * (node) + i for node in fixed_nodes for i in range(degress_of_freedom)]

    for i in range(len(full_mode_vector)):
        if i not in fixed_dofs:
            full_mode_vector[i] = mode_vector[free_dof_idx]
            free_dof_idx += 1

    # Extract translational components
    deformed_coords = original_coords + scale_factor * full_mode_vector[:,:3]

    # Plot original structure (gray)
    for elem in elements:
        node1_idx = elem[0] - 1
        node2_idx = elem[1] - 1
        x = [original_coords[node1_idx, 0], original_coords[node2_idx, 0]]
        y = [original_coords[node1_idx, 1], original_coords[node2_idx, 1]]
        z = [original_coords[node1_idx, 2], original_coords[node2_idx, 2]]
        ax.plot(x, y, z, 'black', alpha=0.9, linewidth=1.5)


    # Sort the list such that deformations are only applied to free nodes
    allowed_deformed = []
    for i, deformed in enumerate(deformed_coords):
        if i in fixed_nodes:
            allowed_deformed.append(original_coords[i])
        else:
            allowed_deformed.append(deformed)
    allowed_deformed = np.array(allowed_deformed)

    # Plot deformed structure (blue)
    for elem in elements:
        node1_idx = elem[0] - 1
        node2_idx = elem[1] - 1
        x = [allowed_deformed[node1_idx, 0], allowed_deformed[node2_idx, 0]]
        y = [allowed_deformed[node1_idx, 1], allowed_deformed[node2_idx, 1]]
        z = [allowed_deformed[node1_idx, 2], allowed_deformed[node2_idx, 2]]
        ax.plot(x, y, z, 'b', linewidth=2)

    # Plot nodes
    ax.scatter(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2],
               c='gray', alpha=0.3, s=30)
    ax.scatter(allowed_deformed[:, 0], allowed_deformed[:, 1], allowed_deformed[:, 2],
               c='blue', s=30)

    # Highlight fixed nodes
    fixed_nodes_coords = original_coords[nodes[:, 2] == 0]
    ax.scatter(fixed_nodes_coords[:, 0], fixed_nodes_coords[:, 1], fixed_nodes_coords[:, 2],
               c='red', s=100, marker='s', label='Fixed supports')

    # Set labels and title
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Mode {mode_number + 1}: {frequency:.2f} Hz')

    # Add legend
    ax.legend(['Original structure', 'Deformed shape', 'Nodes', 'Fixed supports'])

    # Set equal aspect ratio
    ax.set_box_aspect([np.ptp(original_coords[:, 0]),
                       np.ptp(original_coords[:, 1]),
                       np.ptp(original_coords[:, 2])])
    plt.tight_layout()
    return fig

# Appel de la fonction plot_mode_shape pour chaque mode
for mode_number in range(6):
    print(f"Matrix shapes: eigenvectors.shape = {eigenvectors.shape}, frequencies.shape = {frequencies.shape}, elements.shape = {np.array(elemList).shape}, nodes.shape = {np.array(nodeList).shape}")
    mode_vector = eigenvectors[:, mode_number]
    frequency = frequencies[mode_number]
    #fig = plot_mode_shape(nodes=np.array(nodeList), \
                            #elements=np.array(elemList), \
                            #mode_vector=mode_vector, \
                            #mode_number=mode_number, \
                            #frequency=frequency)
    #plt.show()
print("eigenmodes: ", eigenvectors)

# Affichage des matrices de rigidité et de masse globales
print("Matrice de rigidité globale K_global:")
print(K_global)
print("Matrice de masse globale M_global:")
print(M_global)

# Affichage des six premières fréquences naturelles
print("Six premières fréquences naturelles :")
for i in range(6):
    print(f"Fréquence {i+1}: {frequencies[i]:.10f} Hz")

# Effectuer l'étude de convergence
convergence_results = convergence_study()
frequencies_matrix = np.array([result[1] for result in convergence_results])
element_counts = [result[0] for result in convergence_results]
    # Tracer le graphique de l'étude de convergence
plot_convergence_study(element_counts, frequencies_matrix)
print("eigs", eigenvectors)

# Tracer la courbe de convergence
plt.figure(figsize=(8, 6))
plt.plot(divisions, frequencies, marker='o', linestyle='-', label="Fréquence 1")
plt.xlabel("Nombre de divisions (éléments finis)")
plt.ylabel("Fréquence (Hz)")
plt.title("Convergence des fréquences en fonction des divisions")
plt.grid(True)
plt.legend()
plt.show()



def main():
    # Étape 1 – Charger géométrie et matériau
    div_values = [0, 1, 2, 3, 4]
    dofList = create_dof_list(n_nodes: int)
    elem_section_type = define_element_sections(main_beams)
    new_nodes_array, new_beams_array, new_section_type = subdivide_mesh(main_nodes, main_beams, div_values, elem_section_type)
    K_global, M_global = assemble_matrices(new_nodes_array, new_beams_array, new_section_type, E, G, rho)
    M_ass, K_ass = assemble_global_matrices(K_global, M_global, dofList)
    frequencies[:n_modes], eigvecs[:, :n_modes] = extract_modes(M_ass, K_ass, n_modes=6)
    K_global1, M_global1, w, x = Calcul_Eig(M_global, K_global)


if __name__ == "__main__":
    main()



