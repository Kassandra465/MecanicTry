import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import block_diag, eigh
from scipy import linalg

#Material Properties
rho = 7800  # [kg/m**3]
nu = 0.3  # [-]
E = 210 * 1e9  # [Pa]
diametre_exterieur = 0.15  # m
epaisseur_paroi = 0.005  # m

#Section Properties
A = np.pi * (diametre_exterieur**2 - (diametre_exterieur - 2 * epaisseur_paroi)**2) / 4  # [m**2]
I = np.pi * (diametre_exterieur**4 - (diametre_exterieur - 2 * epaisseur_paroi)**4) / 64  # [m**4]
Jx = 2 * I  # [m**4]
G = E / (2 * (1 + nu))  # Shear modulus

div = 0 # Number of beams' divisions (=> (div+1) is the number of elements per beam).

# Initialisation des listes et matrices
elemList = []
dofList = []
nodeList = []

# Définir les nœuds principaux et leurs coordonnées (Structural axis)
main_nodes = np.array([
    [0, 0, 5],      # Nœud 1
    [0, 2, 5],      # Nœud 2
    [1, 0, 4.5],    # Nœud 3
    [1, 2, 4.5],    # Nœud 4
    [2, 0, 4],      # Nœud 5
    [2, 2, 4],      # Nœud 6
    [3, 0, 3.5],    # Nœud 7
    [3, 2, 3.5],    # Nœud 8
    [4, 0, 3],      # Nœud 9
    [4, 2, 3],      # Nœud 10
    [5, 0, 2.5],    # Nœud 11
    [5, 2, 2.5],    # Nœud 12
    [6, 0, 2],      # Nœud 13
    [6, 2, 2],      # Nœud 14
    [7, 0, 1.5],    # Nœud 15
    [7, 2, 1.5],    # Nœud 16
    [8, 0, 1],      # Nœud 17
    [8, 2, 1],      # Nœud 18
    [0, 0, 0],      # Nœud 19
    [0, 2, 0],      # Nœud 20
    [0, 0, 2.5],    # Nœud 21
    [0, 2, 2.5],    # Nœud 22
    [4, 0, 0],      # Nœud 23
    [4, 2, 0],      # Nœud 24
    [4, 0, 1.5],    # Nœud 25
    [4, 2, 1.5],    # Nœud 26
    [8, 0, 0],      # Nœud 27
    [8, 2, 0],      # Nœud 28
    [0, 1, 1.25],   # Nœud 29
    [0, 1, 3.75],   # Nœud 30
    [4, 1, 0.75],   # Nœud 31
    [4, 1, 2.25],   # Nœud 32
    [8, 1, 0.5]     # Nœud 33
])

if (div > 0):
#Definir la discretization FEM
    def generate_dividedNodes_coord(node1, node2, div):
     nodeList_div = []
     for i in range(1, div + 1):
        frac = i / (div + 1)
        new_x = node1[0] + (node2[0] - node1[0]) * frac
        new_y = node1[1] + (node2[1] - node1[1]) * frac
        new_z = node1[2] + (node2[2] - node1[2]) * frac
        nodeList_div.append([new_x, new_y, new_z])

     return nodeList_div

    def create_nodeList_div(elemList):
        nodeList_div = []

        for i in range(len(elemList)):
            current_elem = elemList[i]
            start = current_elem[0]
            end = current_elem[1]

            nodeList_tem = generate_dividedNodes_coord(node1 = nodeList[start - 1],
                                                   node2 = nodeList[end - 1], div=div)
            nodeList_div += nodeList_tem

        return nodeList_div

    
    

# Définir les poutres principales
main_beams = np.array([
    [1, 2],  # Poutre 1
    [1, 3],  # Poutre 2
    [2, 4],  # Poutre 3
    [3, 4],  # Poutre 4
    [3, 5],  # Poutre 5
    [4, 6],  # Poutre 6
    [5, 6],  # Poutre 7
    [5, 7],  # Poutre 8
    [6, 8],  # Poutre 9
    [28, 33],  # Poutre 10
    [7, 8],  # Poutre 11
    [7, 9],  # Poutre 12
    [8, 10], # Poutre 13
    [9, 10], # Poutre 14
    [9, 11], # Poutre 15
    [10, 12], # Poutre 16
    [11, 12], # Poutre 17
    [11, 13], # Poutre 18
    [12, 14], # Poutre 19
    [13, 14], # Poutre 20
    [13, 15], # Poutre 21
    [14, 16], # Poutre 22
    [15, 16], # Poutre 23
    [15, 17], # Poutre 24
    [16, 18], # Poutre 25
    [17, 18], # Poutre 26
    [1, 21],  # Poutre 27
    [1, 30],  # Poutre 28
    [2, 22],  # Poutre 29
    [2, 30],  # Poutre 30
    [21, 30], # Poutre 31
    [22, 30], # Poutre 32
    [21, 22], # Poutre 33
    [20, 22], # Poutre 34
    [21, 29], # Poutre 35
    [22, 29], # Poutre 36
    [19, 29], # Poutre 37
    [20, 29], # Poutre 38
    [21, 19], # Poutre 39
    [9, 32],  # Poutre 40
    [9, 25],  # Poutre 41
    [32, 26], # Poutre 42
    [32, 25], # Poutre 43
    [10, 26], # Poutre 44
    [10, 32], # Poutre 45
    [25, 26], # Poutre 46
    [25, 31], # Poutre 47
    [26, 31], # Poutre 48
    [23, 31], # Poutre 49
    [24, 31], # Poutre 50
    [23, 25], # Poutre 51
    [24, 26], # Poutre 52
    [17, 27], # Poutre 53
    [18, 28], # Poutre 54
    [17, 33], # Poutre 55
    [18, 33], # Poutre 56
    [27, 33], # Poutre 57

])

# Ajouter les nœuds principaux à la liste des nœuds
nodeList.extend(main_nodes)

# Ajouter les poutres principales à la liste des éléments
elemList.extend(main_beams)
if(div > 0):
    nodeList = create_nodeList_div(elemList)

nbr_main_nodes = 33  # Number of nodes of the main structure (without any divisions (i.e. 1 element per beam)).
nbr_final_nodes = nbr_main_nodes + div * len(elemList)  # Total number of nodes for the WHOLE structure.
nbr_dof = 6 * nbr_final_nodes  # *6 because the studied structure is in 3D (6 dofs for each node).

# Définir les degrés de liberté pour les nœuds principaux
def create_dofList(nbr_nodes):
    dofList_final = []

    for i in range(nbr_nodes):
        start_num = i * 6 + 1
        end_num = start_num + 6
        line_sequence = list(range(start_num, end_num))
        dofList_final.append(line_sequence)

    return dofList_final

main_dofs = create_dofList(nbr_nodes=nbr_final_nodes)  # Generating the "dofList" for the WHOLE structure.

nbr_main_elems = len(main_beams)  # Number of elements of the main structure (without any divisions (i.e. 1 element per beam)).
nbr_main_dofs = len(main_dofs)

# Ajouter les degrés de liberté pour les nœuds principaux à la liste des degrés de liberté
dofList.extend(main_dofs)


# Fonction pour calculer la matrice de masse élémentaire (Local axis)
def M_el(rho, A, l):
    r = np.sqrt(I / A)  # Rayon de giration approximatif
    return rho * A * l * np.array([
        [1 / 3, 0, 0, 0, 0, 0, 1 / 6, 0, 0, 0, 0, 0],
        [0, 13 / 35, 0, 0, 0, 11 * l / 210, 0, 9 / 70, 0, 0, 0, -13 * l / 420],
        [0, 0, 13 / 35, 0, -11 * l / 210, 0, 0, 0, 9 / 70, 0, 13 * l / 420, 0],
        [0, 0, 0, r ** 2 / 3, 0, 0, 0, 0, 0, r ** 2 / 6, 0, 0],
        [0, 0, -11 * l / 210, 0, l ** 2 / 105, 0, 0, 0, -13 * l / 420, 0, -l ** 2 / 140, 0],
        [0, 11 * l / 210, 0, 0, 0, l ** 2 / 105, 0, 13 * l / 420, 0, 0, 0, -l ** 2 / 140],
        [1 / 6, 0, 0, 0, 0, 0, 1 / 3, 0, 0, 0, 0, 0],
        [0, 9 / 70, 0, 0, 0, 13 * l / 420, 0, 13 / 35, 0, 0, 0, -11 * l / 210],
        [0, 0, 9 / 70, 0, -13 * l / 420, 0, 0, 0, 13 / 35, 0, 11 * l / 210, 0],
        [0, 0, 0, r ** 2 / 6, 0, 0, 0, 0, 0, r ** 2 / 3, 0, 0],
        [0, 0, 13 * l / 420, 0, -l ** 2 / 140, 0, 0, 0, 11 * l / 210, 0, l ** 2 / 105, 0],
        [0, -13 * l / 420, 0, 0, 0, -l ** 2 / 140, 0, -11 * l / 210, 0, 0, 0, l ** 2 / 105]
    ])

# Fonction pour calculer la matrice de rigidité élémentaire (Local axis)
def K_el(E, G, Jx, A, l, I):
    return np.array([[E*A / l, 0, 0, 0, 0, 0, -E*A / l, 0, 0, 0, 0, 0],
                    [0, 12*E*I / l**3, 0, 0, 0, 6*E*I / l**2, 0, -12*E*I / l**3, 0, 0, 0, 6*E*I / l**2],
                    [0, 0, 12*E*I / l**3, 0, -6*E*I / l**2, 0, 0, 0, -12*E*I / l**3, 0, -6*E*I / l**2, 0],
                    [0, 0, 0, G*Jx / l, 0, 0, 0, 0, 0, -G*Jx / l, 0, 0],
                    [0, 0, -6*E*I / l**2, 0, 4*E*I / l, 0, 0, 0, 6*E*I / l**2, 0, 2*E*I / l, 0],
                    [0, 6*E*I / l**2, 0, 0, 0, 4*E*I / l, 0, -6*E*I / l**2, 0, 0, 0, 2*E*I / l],
                    [-E*A / l, 0, 0, 0, 0, 0, E*A / l, 0, 0, 0, 0, 0],
                    [0, -12*E*I / l**3, 0, 0, 0, -6*E*I / l**2, 0, 12*E*I / l**3, 0, 0, 0, -6*E*I / l**2],
                    [0, 0, -12*E*I / l**3, 0, 6*E*I / l**2, 0, 0, 0, 12*E*I / l**3, 0, 6*E*I / l**2, 0],
                    [0, 0, 0, -G*Jx / l, 0, 0, 0, 0, 0, G*Jx / l, 0, 0],
                    [0, 0, -6*E*I / l**2, 0, 2*E*I / l, 0, 0, 0, 6*E*I / l**2, 0, 4*E*I / l, 0],
                    [0, 6*E*I / l**2, 0, 0, 0, 2*E*I / l, 0, -6*E*I / l**2, 0, 0, 0, 4*E*I / l]])

# Fonction pour obtenir le vecteur de localisation pour un élément
def get_locel(elem_id):
    current_elem = elemList[elem_id]
    start_node = current_elem[0]
    end_node = current_elem[1]
    return dofList[start_node - 1][:] + dofList[end_node - 1][:]

def get_locelnode0(node_id):
    return dofList[node_id][:]

def get_locelnode1(node_id):
    return dofList[node_id][:]

def Rotation_fonction (elem_id):
    node_1 = nodeList[elemList[elem_id][0] - 1]  # Node 1 (Structural axis)
    node_2 = nodeList[elemList[elem_id][1] - 1]  # Node 2 (Structural axis)
    node_3 = [-100, -100, -100]  # Not collinear

    l = np.linalg.norm(np.array(node_2) - np.array(node_1))
    ex = np.array([0.0, 0.0, 0.0])
    for i in range(3):
        ex[i] = (node_2[i] - node_1[i]) / l

    d_2 = np.array([node_2[0] - node_1[0], node_2[1] - node_1[1], node_2[2] - node_1[2]])
    d_3 = np.array([node_3[0] - node_1[0], node_3[1] - node_1[1], node_3[2] - node_1[2]])

    cross_product = np.cross(d_3, d_2)
    ey = cross_product / np.linalg.norm(cross_product)
    ez = np.cross(ey, ex)

    R = np.array([
        [ex[0], ex[1], ex[2]],
        [ey[0], ey[1], ey[2]],
        [ez[0], ez[1], ez[2]]
    ])
    return R

# This function will generate the Global (For the Whole Structure) Mass matrix (M_s) and Stiffness matrix (K_s).
def Generate_Ms_Ks(elem_id):
    total_dof = 196
    
    M_s = np.zeros((total_dof, total_dof))
    K_s = np.zeros((total_dof, total_dof))
   

    # Loop over All the elements of the Whole structure.
    for i in range(len(elemList)):
        current_elem = elemList[i]
        start_node = current_elem[0]
        end_node = current_elem[1]
        node1 = main_nodes[start_node - 1]
        node2 = main_nodes[end_node - 1]
        l = np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2 + (node2[2] - node1[2]) ** 2)
        M_el_ = M_el(rho, A, l)
        K_el_ = K_el(E, G, Jx, A, l, I)

        R = Rotation_fonction(elem_id)
        T = block_diag(R, R, R, R)

        M_e_S = T.T @ M_el_ @ T
        K_e_S = T.T @ K_el_ @ T

        locel_e = get_locel(i)  # locel_e is the locel for the current element.

        

        # Assembly Process of the elementary K_e_S or M_e_S in the Global K_s or M_s.
        for k in range(0, len(locel_e)):
            for s in range(0, len(locel_e)):
                kk = locel_e[k] - 1
                ss = locel_e[s] - 1
                
            
                K_s[kk, ss] += K_e_S[k, s]
                M_s[kk, ss] += M_e_S[k, s]
                

    
    

    return K_s, M_s

# Function to assemble global matrices
def assemble_global_matrices():
    nodes_with_mass = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    total_dof = 196
    M_global1 = np.zeros((total_dof, total_dof))
    K_global1 = np.zeros((total_dof, total_dof))
    total_mass = 0  # For computing the total mass of the structure.

    for elem_id in range(len(elemList)):

        K_s, M_s = Generate_Ms_Ks(elem_id)

        locel = get_locel(elem_id)

        for i in range(12):
            for j in range(12):
                if locel[i] > 0 and locel[j] > 0:
                    M_global1[locel[i] - 1, locel[j] - 1] += M_s[i, j]
                    K_global1[locel[i] - 1, locel[j] - 1] += K_s[i, j]

    loc = []

    locels = np.array(
        [get_locelnode1(18), get_locelnode0(19), get_locelnode0(22), get_locelnode0(23), get_locelnode1(26),
         get_locelnode1(27)])

    loc.extend(locels.flatten())
    for node in nodes_with_mass:
        dof_indices = dofList[6 * node: 6 * node + 2]
        for dof in dof_indices:
            M_global1[dof - 1, dof - 1] += 80 * 51 / 18
    
    # This is used to compute the Total Mass of the Structure, where "u_e" is a full vector of 1 with the same shape of M_el
        # and it represents the Rigid Body Mode (without any deformation, only translation).
    u_e = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    total_mass += np.matmul(np.matmul(u_e.transpose(), M_global1), u_e)
        

    fixed_nodes = [18, 19, 22, 23, 26, 27]  # these nodes are clamped nodes (All their dofs are blocked).
    fixedDof_final = []

    for i in range(len(fixed_nodes)):
        current_node = fixed_nodes[i]

        tem = dofList[current_node - 1][:]
        fixedDof_final += tem

    return fixedDof_final

    # Apply the Fixed BCs.
    fixed_dof = create_fixedDof()

    rows_to_delete = fixed_dof
    columns_to_delete = fixed_dof
    # Deleting the specified rows and columns of the K_s and M_s.
    K_global1 = np.delete(np.delete(K_global1, rows_to_delete, axis=0), columns_to_delete, axis=1)
    M_global1 = np.delete(np.delete(M_global1, rows_to_delete, axis=0), columns_to_delete, axis=1)

    return M_global1, K_global1, total_mass

M, K, mass = assemble_global_matrices()
print("la masse totale est:", mass)
##### Applying the Boundary Conditions (BCs)


# Fonction pour extraire les fréquences naturelles et les modes associés
def extract_frequencies(K_global, M_global):
    eigenvalues, eigenvectors = eigh(K_global, M_global)
    frequencies = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
    return frequencies, eigenvectors

# Fonction pour effectuer l'étude de convergence
def convergence_study():
    num_elements_per_beam = div
    convergence_results = []

    for num_elem in num_elements_per_beam:

        # Assemblage des matrices globales
        M_global, K_global = assemble_global_matrices()

        # Extraction des fréquences naturelles et des modes associés
        frequencies, eigenvectors = extract_frequencies(K_global, M_global)

        # Stockage des résultats
        convergence_results.append((num_elem, frequencies[:6]))

    return convergence_results

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

#       Draw Points Fct      #

def Draw_Structure(main_dofs, mode):
    # Define the 3D points.
    points2 = main_dofs
    # Define the connections between points.
    connections_structure = elemList
    # Extract x, y, and z coordinates.
    x2, y2, z2 = zip(*points2)

    # Create a 3D figure.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points.
    ax.scatter(x2, y2, z2, s=3, c='k', label='Points')

    # Plot the connections.
    for connection in connections_structure:
        point2_1 = points2[connection[0] - 1]
        point2_2 = points2[connection[1] - 1]
        ax.plot([point2_1[0], point2_2[0]], [point2_1[1], point2_2[1]], [point2_1[2], point2_2[2]], 'b-')

    # Set axes.
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim(-1, 8.5)
    ax.set_ylim(-1, 8.5)
    ax.set_zlim(0, 6)
    ax.set_box_aspect([1, 1, 2])  # Stretch the z-axis by a factor of 2

    ax.set_title(f'Mode {mode}')
    ax.invert_xaxis()
    plt.show()

#        Calcul Eig + shape

def Calcul_Eig(M_global1, K_global1, draw=True):
    # Calcul Eigen values and vectors (Shapes).
    w_carre, eigenModes = linalg.eig(K_global1, M_global1)
    w_carre = np.array(w_carre)  # np.array is used for doing some operations.
    w = np.sqrt(w_carre)  # Pay attention that this w is in [rad/s].

    eigenFreq = w / (2 * math.pi)  # For getting the normal frequencies in [Hz].

    sorted_indices = np.argsort(eigenFreq)  # Sort eigenFreq and get the indices for sorting.
    sorted_eigenFreq = eigenFreq[sorted_indices]

    sorted_eigenModes = eigenModes[:, sorted_indices]  # Rearrange columns of eigenModes based on sorted indices.

    print(f'\n##################### sorted Eigen/Normal Frequencies (div = {div}): #####################')
    print(sorted_eigenFreq[:6].real)

    return M_global1, K_global1, sorted_eigenFreq, sorted_eigenModes

def draw_EigenModes(x):
    nbr_modes = 6  # number of normal modes to draw (deformed structure).
    for mode in range(1, nbr_modes + 1):
        new_nodes_coords = []

        for i in range(0, int(len(x[0]) / 6)):
            current_eigenMode = x[:, mode - 1]
            current_dofs = current_eigenMode[i * 6: (i * 6) + 6]
            coord_xyz = current_dofs[:3]
            mul = 10  # is the multiplication factor (amplification) to see *better* the normal modes with the graphs.
            new_x = mul * coord_xyz[0].real + nodeList[i][0]
            new_y = mul * coord_xyz[1].real + nodeList[i][1]
            new_z = mul * coord_xyz[2].real + nodeList[i][2]
            new_nodes_coords.append([new_x, new_y, new_z])

        # Draw each mode.
        Draw_Structure(new_nodes_coords=new_nodes_coords, mode=mode)

#==================#
#       Main       #
#==================#
M_global, K_global = assemble_global_matrices()  # Assemblage des matrices globales
K_global1, M_global1, w, x = Calcul_Eig(M_global, K_global)

frequencies, eigenvectors = extract_frequencies(K_global, M_global)  # Extraction des fréquences naturelles et des modes associés
draw_EigenModes(x=x)  # Un/comment for drawing the 6 shape modes.

# Affichage des six premières fréquences naturelles
print("Six premières fréquences naturelles :")
for i in range(6):
    print(f"Fréquence {i+1}: {frequencies[i]:.2f} Hz")
