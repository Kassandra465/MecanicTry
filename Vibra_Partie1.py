import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, eigh

# ==========================
# Matériaux et propriétés
# ==========================
rho = 7800.0           # kg/m3
E = 210e9              # Pa
nu = 0.3
G = E / (2*(1+nu))     # Module de cisaillement

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
# Section
# ==========================
def section(D, t):
    Di = D - 2 * t
    A = np.pi * (D**2 - Di**2) / 4.0
    Iy = np.pi * (D**4 - Di**4) / 64.0
    Iz = Iy
    Jx = 2 * Iy
    return A, Iy, Iz, Jx


def define_element_sections(main_beams): #créé un dictionaire pour attribuer les bonnes sections au bon endroit après division
    # Cadres rouges
    A_f, Iy_f, Iz_f, Jx_f = section(0.120, 0.005)
    # Poutres bleues
    A_s, Iy_s, Iz_s, Jx_s = section(0.070, 0.003)

    elem_section_type = {}

    for i in range(len(main_beams)):
        if i < 22:
            elem_section_type[i] = {
                "type": "frame",
                "A": A_f, "Iy": Iy_f, "Iz": Iz_f, "Jx": Jx_f
            }
        else:
            elem_section_type[i] = {
                "type": "support",
                "A": A_s, "Iy": Iy_s, "Iz": Iz_s, "Jx": Jx_s
            }

    return elem_section_type

# -----------------------
# Calcul masse totale (poutres + toit)
# -----------------------
def compute_total_mass(nodes, beams, elem_section_type, rho):
    roof_mass = 500.0
    structure_mass = 0.0
    for i, (n1, n2) in enumerate(beams):
        coord1 = np.array(nodes[n1])
        coord2 = np.array(nodes[n2])
        L = np.linalg.norm(coord2 - coord1)
        A = elem_section_type[i]["A"]
        structure_mass += rho * A * L
    total_mass = structure_mass + roof_mass
    print("\n============== MASSE TOTALE ==============")
    print(f"Masse des poutres : {structure_mass:.2f} kg")
    print(f"Masse concentrée du toit : {roof_mass:.2f} kg")
    print(f"Masse totale de la structure : {total_mass:.2f} kg")
    print("==========================================")
    return total_mass

# ==========================
# Division en elements per beam
# ==========================
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

        # Identifie le type de section
        beam_type = elem_section_type.get(i, 'frame')

        previous_node = node1

        # Créé les noeuds intermédiaire
        for k in range(1, div + 1):
            new_coord = coord1 + k * (coord2 - coord1) / (div + 1)
            new_nodes.append(new_coord)

            # créé les subdivisions
            new_beams.append([previous_node, current_node_index])
            new_section_type[new_elem_index] = beam_type
            new_elem_index += 1

            previous_node = current_node_index
            current_node_index += 1

        # Connecte le dernier noeud
        new_beams.append([previous_node, node2])
        new_section_type[new_elem_index] = beam_type
        new_elem_index += 1

    # Combine tous les noeuds
    new_nodes_array = np.vstack((main_nodes, np.array(new_nodes)))
    new_beams_array = np.array(new_beams, dtype=int)

    return new_nodes_array, new_beams_array, new_section_type

# -----------------------
# Degrés de liberté (6 par noeud)
# -----------------------
def create_dof_list(n_nodes: int):
    dof_list = []
    for i in range(n_nodes):
        node_dofs = list(range(i * 6, i * 6 + 6))
        dof_list.append(node_dofs)

    return dof_list

# ==========================
# Matrice de rotation
# ==========================
def Rotation_fonction(node1, node2):
    x_axis = node2 - node1
    L = np.linalg.norm(x_axis)
    if L < 1e-9:
        raise ValueError("Longueur d'élément nulle")
    ex = x_axis / L
    ey = np.cross([0, 0, 1], ex)
    if np.linalg.norm(ey) < 1e-6:
        ey = np.cross([0, 1, 0], ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R = np.vstack((ex, ey, ez))
    return R

#=========================
# Matrices élémentaires
# ==========================
def K_el(E, G, A, l, Iy, Iz, Jx):
    return np.array([
        [ E*A/l, 0, 0, 0, 0, 0, -E*A/l, 0, 0, 0, 0, 0],
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

        # section properties pour cet élément
        props = elem_section_dict[e]
        A = props['A']; Iy = props['Iy']; Iz = props['Iz']; Jx = props['Jx']

        # matrices locales
        Ke_local = K_el(E, G, A, L, Iy, Iz, Jx)
        Me_local = M_el(rho, A, L, Iy, Iz)

        # rotation et transformation
        R = Rotation_fonction(node1, node2)   # gère inclinaison en 3D
        T = block_diag(R, R, R, R)  # 12x12

        Ke_g = T.T @ Ke_local @ T
        Me_g = T.T @ Me_local @ T

        # assemblage dans K_global/M_global
        loc = dof_list[n1] + dof_list[n2]   # 12 indices
        for i_loc in range(12):
            for j_loc in range(12):
                K_global[loc[i_loc], loc[j_loc]] += Ke_g[i_loc, j_loc]
                M_global[loc[i_loc], loc[j_loc]] += Me_g[i_loc, j_loc]

    return K_global, M_global


def assemble_global_matrices(K_s, M_s, dofList):
    nodes_with_mass = [5, 6, 7, 8, 16, 17, 18, 19] # masses du toit
    roof_mass = 500.0 / 8
    M_global = M_s.copy()
    K_global = K_s.copy()

    # Ajout des masses concentrées sur les DDL de translation
    for nd in nodes_with_mass:
        for dof in dofList[nd][:3]:  # ux, uy, uz uniquement
            M_global[dof, dof] += roof_mass

    # Application des encastrements
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
    # Vérification
    if np.any(np.isnan(K_reduced)) or np.any(np.isnan(M_reduced)):
        raise ValueError("Matrices K ou M contiennent des élément mal défini.")
    if np.any(np.linalg.eigvals(M_reduced) <= 0):
        raise ValueError(
            "La matrice de masse M n’est pas positive définie !")

    # Résolution
    eigvals, eigvecs = eigh(K_reduced, M_reduced)
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
        norm = np.sqrt(mode.T @ M_reduced @ mode)
        if norm > 0:
            eigvecs[:, i] = mode / norm

    return frequencies[:n_modes], eigvecs[:, :n_modes]

# ==========================
# Etude de convergence
# ==========================
def convergence_study(main_beams, main_nodes, E, G, rho):
    div_values = [0, 1, 2, 3, 4, 5]
    elem_section_type = define_element_sections(main_beams)
    convergence_results = []
    for i in div_values:
        # Assemblage
        new_nodes_array, new_beams_array, new_section_type = subdivide_mesh(main_nodes, main_beams, i, elem_section_type)
        dofList = create_dof_list(len(new_nodes_array))
        M_global, K_global = assemble_matrices(new_nodes_array, new_beams_array, new_section_type, E, G, rho)
        M_ass, K_ass = assemble_global_matrices(M_global, K_global, dofList)

        # Extraction
        frequencies, _ = extract_modes(K_ass, M_ass)

        # Stockage
        convergence_results.append((i, frequencies[:6]))

    return convergence_results


#==================#
#       Main       #
#==================#
# Fonction pour tracer l'étude de convergence
def plot_convergence_study(convergence_results, n_modes=6):
    divs = [r[0] for r in convergence_results]
    freq_matrix = np.array([r[1][:n_modes] for r in convergence_results])

    plt.figure(figsize=(10, 6))
    for i in range(n_modes):
        plt.plot(divs, freq_matrix[:, i], marker='o', label=f'Mode {i + 1}')
    plt.xlabel('Nombre de divisions (div)')
    plt.ylabel('Fréquence naturelle [Hz]')
    plt.title('Étude de convergence des fréquences naturelles')
    plt.legend()
    plt.grid(True)
    plt.show()

#Fonction pour tracer les modes shapes avec la structure originale
def plot_mode_shape(nodes, elements, mode_vector, fixed_nodes, mode_number, frequency,scale_factor=20.0):
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    mode_vector = np.real(np.asarray(mode_vector).flatten())

    n_nodes = nodes.shape[0]
    dof_per_node = 6

    # Reconstruction du vecteur modal complet
    full_mode = np.zeros((n_nodes, dof_per_node))
    free_idx = 0
    for nd in range(n_nodes):
        if nd in fixed_nodes:
            continue
        for k in range(dof_per_node):
            if free_idx < mode_vector.size:
                full_mode[nd, k] = mode_vector[free_idx]
                free_idx += 1

    # Déformée (translations uniquement)
    displacements = full_mode[:, :3]
    if scale_factor is None:
        max_disp = np.max(np.linalg.norm(displacements, axis=1))
        coord_range = np.max(np.ptp(nodes, axis=0))
        scale_factor = 0.1 * coord_range / max_disp if max_disp > 0 else 1.0
    deformed = nodes + scale_factor * displacements

    # Tracé
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Barres originales et déformées
    for (n1, n2) in elements:
        x0, y0, z0 = nodes[n1]; x1, y1, z1 = nodes[n2]
        xd0, yd0, zd0 = deformed[n1]; xd1, yd1, zd1 = deformed[n2]

        # Structure originale (gris clair)
        ax.plot([x0, x1], [y0, y1], [z0, z1],
                color='0.6', alpha=0.4, linewidth=1.2)
        # Structure déformée (bleu)
        ax.plot([xd0, xd1], [yd0, yd1], [zd0, zd1],
                color='C0', linewidth=2.0)

    # Points avec masses concentrées (verts)
    nodes_with_mass = [5, 6, 7, 8, 16, 17, 18, 19]
    if nodes_with_mass is not None and len(nodes_with_mass) > 0:
        nm = np.array(nodes_with_mass, dtype=int)
        ax.scatter(nodes[nm, 0], nodes[nm, 1], nodes[nm, 2],
                   c='limegreen', s=90, marker='o', edgecolors='black',
                   label='Mass nodes')

    # Nœuds fixes (rouges)
    if len(fixed_nodes) > 0:
        fn = np.array(fixed_nodes, dtype=int)
        ax.scatter(nodes[fn, 0], nodes[fn, 1], nodes[fn, 2],
                   c='red', s=90, marker='s', label='Fixed supports')

    # Échelle homogène
    all_coords = np.vstack((nodes, deformed))
    mid = np.mean(all_coords, axis=0)
    span = np.ptp(all_coords, axis=0)
    max_range = np.max(span)
    for axis, mid_val in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(mid_val - max_range / 2, mid_val + max_range / 2)

    # Légende
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f"Mode {mode_number+1} — {frequency:.3f} Hz (scale={scale_factor:.2g})")
    ax.view_init(elev=25, azim=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig


def main():
    # Étude de convergence
    convergence_results= convergence_study(main_beams, main_nodes, E, G, rho)
    plot_convergence_study(convergence_results)

    # Calcul final pour div 4 (après étude de convergence)
    div = 4
    elem_section_type = define_element_sections(main_beams)
    new_nodes, new_beams, section_type = subdivide_mesh(main_nodes, main_beams, div, elem_section_type)
    dofList = create_dof_list(len(new_nodes))

    # Assemblage global
    M_global, K_global = assemble_matrices(new_nodes, new_beams, section_type, E, G, rho)
    M_ass, K_ass = assemble_global_matrices(M_global, K_global, dofList)

    # Extraction modale
    frequencies, eigvecs = extract_modes(K_ass, M_ass, n_modes=6)

    #Affichage des résultats
    print("\n=================== FREQUENCES PROPRES ===================")
    for i, f in enumerate(frequencies[:6]):
        print(f"Mode {i+1} : {f:.4f} Hz")

    print("\n=================== VECTEURS PROPRES =====================")
    print(eigvecs[:, :6])


    print("\n=================== MATRICES GLOBALES ===================")
    print("Matrice de masse globale M_ass :")
    print(M_ass)
    print("\nMatrice de rigidité globale K_ass :")
    print(K_ass)

    # Visualisation des modes
    fixed_nodes = [5, 10, 16, 21]
    for mode_number in range(6):
        mode_vector = eigvecs[:, mode_number]
        frequency = frequencies[mode_number]
        plot_mode_shape(
            nodes=np.array(new_nodes),
            elements=new_beams,
            mode_vector=mode_vector,
            fixed_nodes=fixed_nodes,
            mode_number=mode_number,
            frequency=frequency
        )

    plt.show()


if __name__ == "__main__":
    main()



