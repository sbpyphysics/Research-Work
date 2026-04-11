import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity


############### PARAMETERS AND HAMILTONIAN NEEDED



H = H.tocsc()
I = identity(N, dtype=np.complex128, format='csc')

eng = np.linspace(-6,6,500)
dos = np.zeros_like(eng, dtype=float)



E = eigvalsh(H.toarray())

# DOS
dos = np.zeros_like(eng)
for i, e in enumerate(eng):
    dos[i] = np.sum(eta / ((e - E)**2 + eta**2)) / (np.pi * N)



plt.figure(figsize=(12, 8))

plt.plot(eng, dos, linewidth=3.5, color='red')

# Axis and text styling
plt.xlabel("E", fontname='Times New Roman', fontsize=26, color='black')
plt.ylabel(r'$\rho$', fontname='Times New Roman', fontsize=26, color='black')


# Tick label styling
plt.xticks(fontname='Times New Roman', fontsize=26, color='black')
plt.yticks(fontname='Times New Roman', fontsize=26, color='black')
plt.ylim(0,1)

# --- Add grid, symmetry line, legend ---
plt.grid(True)
plt.axvline(0, color='gray', linestyle=':', lw=1.5, alpha=0.7)

plt.tight_layout()
plt.show()

#%%
energies = np.linspace(-3, 3, 1200)

############### TRANSPORT CALCULATION #########################
Tlist = []


def surface_gf_1D(E, t):
    z = E + 1e-12j   # ensures correct analytic continuation
    return (z - np.sqrt(z**2 - 4*t**2)) / (2 * t**2)


def self_energy_1D(E, N, site, t_lead=-1.0, t_c=-1.0):
    Sigma = np.zeros((N, N), dtype=complex)
    g = surface_gf_1D(E, t_lead)
    Sigma[site, site] = (t_c**2) * g
    return Sigma


def transmission(H, energies, left_site, right_site,
                 t_lead=-1.0, t_c=-1.0, eta=1e-6):

    N = H.shape[0]
    Tlist = []

    I = np.eye(N, dtype=complex)

    for E in energies:

        SigmaL = self_energy_1D(E, N, left_site, t_lead, t_c)
        SigmaR = self_energy_1D(E, N, right_site, t_lead, t_c)

        GammaL = 1j * (SigmaL - SigmaL.conj().T)
        GammaR = 1j * (SigmaR - SigmaR.conj().T)

        G = np.linalg.solve((E + 1j*eta)*I - H - SigmaL - SigmaR, I)

        # Fast formula (since single-site coupling)
        T = np.real(GammaL[left_site, left_site] *
                    abs(G[left_site, right_site])**2 *
                    GammaR[right_site, right_site])

        Tlist.append(T)

    return np.array(Tlist)

H = H.toarray()
left_site = 0
right_site = m - 1
T = transmission(H, energies, left_site, right_site)

# Plot
plt.figure(figsize=(12,8))
plt.plot(energies, T, lw=2)
plt.xlabel("Energy",  fontsize=26, color='black')
plt.ylabel("Transmission", fontsize=26, color='black')

plt.xticks(fontname='Times New Roman', fontsize=26, color='black')
plt.yticks(fontname='Times New Roman', fontsize=26, color='black')
plt.ylim(0,1)


plt.grid(True)
plt.axvline(0, color='gray', linestyle=':', lw=1.5, alpha=0.7)

plt.tight_layout()
plt.show()
#%%
##ANOTHER VERSION OF TRANSPORT

############### TRANSPORT CALCULATION #########################

energies = np.linspace(-3, 3, 1200)
Tlist = []

'''
def surface_gf_1D(E, t):
    z = E + 1e-12j   # ensures correct analytic continuation
    return (z - np.sqrt(z**2 - 4*t**2)) / (2 * t**2)
'''

def surface_gf_1D(E, t):
    z = E + 1e-12j
    root = np.sqrt(z**2 - 4*t**2)
    return (z - root) / (2 * t**2)

def self_energy_1D(E, N, site, t_lead=-1.0, t_c=-1.0):
    Sigma = np.zeros((N, N), dtype=complex)
    g = surface_gf_1D(E, t_lead)
    Sigma[site, site] = (t_c**2) * g
    return Sigma

'''
def transmission(H, energies, left_site, right_site,
                 t_lead=-1.0, t_c=-1.0, eta=1e-6):

    N = H.shape[0]
    Tlist = []

    I = identity(N, dtype=complex)

    for E in energies:

        SigmaL = self_energy_1D(E, N, left_site, t_lead, t_c)
        SigmaR = self_energy_1D(E, N, right_site, t_lead, t_c)

        GammaL = 1j * (SigmaL - SigmaL.conj().T)
        GammaR = 1j * (SigmaR - SigmaR.conj().T)

        G = spsolve((E + 1j*eta)*I - H - SigmaL - SigmaR, I)

        # Fast formula (since single-site coupling)
        T = np.real(GammaL[left_site, left_site] *
                    abs(G[left_site, right_site])**2 *
                    GammaR[right_site, right_site])

        Tlist.append(T)

    return np.array(Tlist)
'''
def transmission(H, energies, left_site, right_site,
                 t_lead=-1.0, t_c=-1.0, eta=1e-6):

    N = H.shape[0]
    Tlist = []

    I = identity(N, dtype=complex)

    for E in energies:

        SigmaL = self_energy_1D(E, N, left_site, t_lead, t_c)
        SigmaR = self_energy_1D(E, N, right_site, t_lead, t_c)

        GammaL = 1j * (SigmaL - SigmaL.conj().T)
        GammaR = 1j * (SigmaR - SigmaR.conj().T)

        A = (E + 1j*eta)*I - H - SigmaL - SigmaR

        # solve only needed column
        b = np.zeros(N, dtype=complex)
        b[right_site] = 1.0

        G_col = spsolve(A, b)

        T = np.real(
            GammaL[left_site, left_site] *
            abs(G_col[left_site])**2 *
            GammaR[right_site, right_site]
        )

        Tlist.append(T)

    return np.array(Tlist)

H = H.toarray()
left_site = 0
right_site = N - 1
T = transmission(H, energies, left_site, right_site)

# Plot
plt.figure(figsize=(12,8))
plt.plot(energies, T, lw=2)
plt.xlabel("Energy",  fontsize=26, color='black',fontname='Times New Roman')
plt.ylabel("Transmission", fontsize=26, color='black',fontname='Times New Roman')

plt.xticks(fontname='Times New Roman', fontsize=26, color='black')
plt.yticks(fontname='Times New Roman', fontsize=26, color='black')
plt.ylim(0,1)


plt.grid(True)
plt.axvline(0, color='gray', linestyle=':', lw=1.5, alpha=0.7)

plt.tight_layout()
plt.show()
#%%
############ FLAT BAND CALCULATION CODE ##########################
import numpy as np
import matplotlib.pyplot as plt


def build_k_hamiltonian(k, H0, T, a=1.0):
    return H0 + T * np.exp(1j * k * a) + T.conj().T * np.exp(-1j * k * a)


# -------------------------------
# Band structure
# -------------------------------
def compute_band_structure(H0, T, k_vals, a=1.0):
    n = H0.shape[0]
    bands = np.zeros((len(k_vals), n))
    
    for idx, k in enumerate(k_vals):
        Hk = build_k_hamiltonian(k, H0, T, a)
        bands[idx] = np.linalg.eigvalsh(Hk)
    
    return bands


# -------------------------------
# Plotting
# -------------------------------
def plot_bands(k_vals, bands):
    plt.figure(figsize=(6,4))
    
    for i in range(bands.shape[1]):
        plt.plot(k_vals, bands[:, i], color='black')
    
    plt.xlabel(r"$k$")
    plt.ylabel("Energy")
    plt.title("Band Structure")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
#%%