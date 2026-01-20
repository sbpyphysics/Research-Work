import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity #Sparse matrix for ease of ram usage
from scipy.sparse.linalg import eigsh #Inbuilt function for the eigen value and vector calcs


#Physical Parameters
h= 1   
m = 1         
Lx, Ly = 5,3   

Nx, Ny = 160,80  # Number of grid points 
dx, dy = Lx/(Nx+1), Ly/(Ny+1)  # Grid spacing
N_states = int(input("How many states have to calculate: "))  # Number of eigenstates

#For x axis
main_x = -2.0 * np.ones(Nx) #an array of 160 points of Nx
off_x = np.ones(Nx - 1) #Off diagonal so one point less
Lx_mat = diags([off_x, main_x, off_x], [-1, 0, 1]) / dx**2 #Making the matrix 

#For y axis
main_y = -2.0 * np.ones(Ny)
off_y = np.ones(Ny - 1)
Ly_mat = diags([off_y, main_y, off_y], [-1, 0, 1]) / dy**2


Laplacian_2D = kron(identity(Ny), Lx_mat) + kron(Ly_mat, identity(Nx))


H = -(h**2 / (8 *np.pi**2* m)) * Laplacian_2D


eigvals, eigvecs = eigsh(H, k=N_states, which='SM')  # print eigenvalues from Smallest to highest
eigvals = np.real(eigvals)

#Eigen Values
for i, E in enumerate(eigvals):
    print(f"State {i+1}: E = {E:.4f}")



IPRs = []  #Empty list to store the IPR values

#Plotting
X = np.linspace(0, Lx, Nx+2)
Y = np.linspace(0, Ly, Ny+2)

for i in range(N_states):
    psi = np.zeros((Ny+2, Nx+2))
    psi[1:-1, 1:-1] = eigvecs[:, i].reshape((Ny, Nx))
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)

    prob_density = np.abs(psi)**2
    ipr = np.sum(prob_density**2.0)*dx*dy
    IPRs.append(ipr)

    plt.figure(figsize=(10, 7))
    plt.imshow(prob_density, extent=[0, Lx*1e9, 0, Ly*1e9],
               origin='lower', cmap='RdPu', aspect='auto')
    plt.colorbar(label=r'$|\psi(x,y)|^2$')
    plt.title(f"State {i+1}, E = {eigvals[i]:.4f} eV")
    plt.xlabel("x (nm)")
    plt.ylabel("y (nm)")
    plt.tight_layout()
    plt.show()
    


print("\n=== Inverse Participation Ratios (IPRs) ===") #IPR is constant for 2d rectangular potential well
for k, ipr in enumerate(IPRs):
    print(f"State {k+1:2d}: IPR = {ipr:.6e}")
plt.plot(range(1, N_states + 1), IPRs, 'o-', color='crimson', lw=1.5, markersize=6)
plt.xlabel("Eigenstate Index")
plt.ylabel("Inverse Participation Ratio (IPR)")
plt.title("IPR vs Eigenstate Index for Rectangular Potential well")
plt.grid(True, alpha=0.4, color='k')
plt.tight_layout()
plt.show()




