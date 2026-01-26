####install block2
####this script is for electron-phonon coupling strength of 0.1, change the G value for different coupling strengths
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import numpy as np

N_SITES_ELEC, N_SITES_PH, N_ELEC = 8, 8, 8
N_PH, OMEGA, G = 11, 0.25, 0.1
L = N_SITES_ELEC + N_SITES_PH

U_LIST = [0.2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=L, n_elec=N_ELEC, spin=0)

# [Part A] Set states and matrix representation of operators in local Hilbert space
site_basis, site_ops = [], []
Q = driver.bw.SX # quantum number wrapper (n_elec, 2 * spin, point group irrep)

for k in range(L):
    if k < N_SITES_ELEC:
        # electron part
        basis = [(Q(0, 0, 0), 1), (Q(1, 1, 0), 1), (Q(1, -1, 0), 1), (Q(2, 0, 0), 1)] # [0ab2]
        ops = {
            "": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),   # identity
            "c": np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),  # alpha+
            "d": np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),  # alpha
            "C": np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0]]), # beta+
            "D": np.array([[0, 0, 1, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]), # beta
        }
    else:
        # phonon part
        basis = [(Q(0, 0, 0), N_PH)]
        ops = {
            "": np.identity(N_PH), # identity
            "E": np.diag(np.sqrt(np.arange(1, N_PH)), k=-1), # ph+
            "F": np.diag(np.sqrt(np.arange(1, N_PH)), k=1),  # ph
        }
    site_basis.append(basis)
    site_ops.append(ops)

# [Part B] Set Hamiltonian terms in Hubbard-Holstein model
driver.ghamil = driver.get_custom_hamiltonian(site_basis, site_ops)

for U in U_LIST:
    b = driver.expr_builder()

    # electron part
    b.add_term("cd", np.array([[i, i + 1, i + 1, i] for i in range(N_SITES_ELEC - 1)]).ravel(), -1)
    b.add_term("CD", np.array([[i, i + 1, i + 1, i] for i in range(N_SITES_ELEC - 1)]).ravel(), -1)
    b.add_term("cdCD", np.array([[i, i, i, i] for i in range(N_SITES_ELEC)]).ravel(), U)

    # phonon part
    b.add_term("EF", np.array([[i + N_SITES_ELEC, ] * 2 for i in range(N_SITES_PH)]).ravel(), OMEGA)

    # interaction part
    b.add_term("cdE", np.array([[i, i, i + N_SITES_ELEC] for i in range(N_SITES_ELEC)]).ravel(), G)
    b.add_term("cdF", np.array([[i, i, i + N_SITES_ELEC] for i in range(N_SITES_ELEC)]).ravel(), G)
    b.add_term("CDE", np.array([[i, i, i + N_SITES_ELEC] for i in range(N_SITES_ELEC)]).ravel(), G)
    b.add_term("CDF", np.array([[i, i, i + N_SITES_ELEC] for i in range(N_SITES_ELEC)]).ravel(), G)

    # [Part C] Perform DMRG
    mpo = driver.get_mpo(b.finalize(adjust_order=True, fermionic_ops="cdCD"), algo_type=MPOAlgorithmTypes.FastBipartite)
    mps = driver.get_random_mps(tag=f"KET_U{U}", bond_dim=250, nroots=1)
    energy = driver.dmrg(mpo, mps, n_sweeps=10, bond_dims=[250] * 4 + [500] * 4,
        noises=[1e-4] * 4 + [1e-5] * 4 + [0], thrds=[1e-10] * 8, dav_max_iter=30, iprint=1)

    print("U = %s  DMRG energy = %20.15f" % (str(U), energy))
