import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

# ---------------- PARAMETERS ----------------
N = 200          # number of sites
t = 1.0          # hopping

# ---------------- HAMILTONIAN ----------------
H = lil_matrix((N, N), dtype=np.float64)

for i in range(N - 1):
    H[i, i+1] = -t
    H[i+1, i] = -t

# ---------------- EIGENVALUES ----------------
H_dense = H.toarray()
eigenvalues, eigenvectors = np.linalg.eigh(H_dense)

# ---------------- IPR ----------------
energy_values = []
I_values = []

for idx in range(len(eigenvalues)):
    psi = eigenvectors[:, idx]
    I = np.sum(np.abs(psi)**4)
    
    energy_values.append(eigenvalues[idx])
    I_values.append(I)

# ---------------- PLOT ----------------
plt.figure(figsize=(12, 8))

plt.scatter(energy_values, I_values, color='black', s=15)

plt.xlabel('Energy E', fontname='Times New Roman', fontsize=26)
plt.ylabel('IPR', fontname='Times New Roman', fontsize=26)

plt.xticks(fontname='Times New Roman', fontsize=20)
plt.yticks(fontname='Times New Roman', fontsize=20)

plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()