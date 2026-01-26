###for a list of U values
import pathlib
import numpy as np
from pyscf import gto, scf, fci, ao2mo, mcscf


from time import perf_counter

from molbe import BE, fragpart


# ---Hubbard-Holstein Model Parameters ---
n_sites = 8  # Number of sites in the chain

U_LIST = [0.2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # hubbard onsite interaction U values

omega0 = 0.25  # Phonon frequency
g_coupling = 0.1 # electron-phonon coupling strength

# Define the string for the n_sites hydrogen atoms, we will later substract the nuclear repulsion energy to make it compatible with lattic sites
spacing_value = 1  # Distance between hydrogen atoms
atom_string_for_mol = "".join([f"H 0 0 {m*spacing_value};" for m in range(n_sites)]).rstrip(';')

# Create the pyscf.gto.Mole object for the system
mol_hub = gto.M(
    atom=atom_string_for_mol,
    basis="STO-3G",
    charge=0,
    spin=0,  # Assuming a singlet ground state 
)
mol_hub.nelectron = 4 # number of electrons
mol_hub.incore_anyway = True  # Ensure integrals are handled in memory

# --- Parameters for Inner Hubbard-Holstein SCF Loop (Electronic-Phonon coupling) ---
max_inner_hh_iter = 1000  # Maximum iterations for the electronic-phonon self-consistency (inner loop)
inner_hh_alpha_conv_tol = 1.0e-5  # Convergence tolerance for alpha values in inner loop

# --- Parameters for Grand Outer Self-Consistency Loop (BE-coupled) ---
max_grand_outer_iter = 20  # Max iterations for the full BE-coupled self-consistency
grand_alpha_conv_tol = 5.0e-7  # Convergence tolerance for alpha from BE RDM1


# ============================
# Sweep over U values
# ============================
final_energy_by_U = {}

for U in U_LIST:
    print("\n" + "#" * 90)
    print(f"#  Starting U sweep point: U = {U}")
    print("#" * 90)

    # Set Hubbard interaction for this sweep point
    u_interaction = U

    # Initial guess for alpha values for the very first grand iteration (per U)
    real_part = np.random.rand(n_sites)
    imag_part = np.random.rand(n_sites)
    alpha_values_for_grand_loop = real_part + 1j * imag_part

    grand_total_energy_history = []  # per-U history

    print("\n--- Starting Grand Self-Consistent Field (SCF) Loop (BE-coupled Hubbard-Holstein) ---")

    start_time_total_U = perf_counter()  # total timer per U

    # --- Grand Outer Self-Consistency Loop ---
    for grand_outer_iter in range(max_grand_outer_iter):
        print(f"\n======== Grand SCF Iteration: {grand_outer_iter + 1} (U = {U}) ========")
        old_alpha_values_for_grand_loop = alpha_values_for_grand_loop.copy()
        print(f"  Alpha values at start of Grand Iteration {grand_outer_iter + 1}: {alpha_values_for_grand_loop.real}")

        # --- Initialize PySCF MF object for this grand iteration ---
        mf = scf.RHF(mol_hub)
        mf.get_ovlp = lambda *args: np.eye(n_sites)

        # Initialize eri as a standard 4D numpy array, populate with Hubbard U terms
        eri_hubbard = np.zeros([n_sites] * 4, dtype=np.float64)
        for i in range(n_sites):
            eri_hubbard[i, i, i, i] = u_interaction
        mf._eri = ao2mo.restore(8, eri_hubbard, n_sites)
        mf.init_guess = '1e'  # Set initial guess for SCF

        # --- Inner Hubbard-Holstein SCF Loop (Electronic-Phonon Coupling) ---
        inner_hh_alpha_values = alpha_values_for_grand_loop.copy()

        print(f"\n--- Inner Hubbard-Holstein SCF (Electronic-Phonon Coupling) ---")
        for inner_iter in range(max_inner_hh_iter):
            old_inner_hh_alpha_values = inner_hh_alpha_values.copy()

            # Construct the effective one-electron Hamiltonian (h1_effective)
            h1_effective = np.zeros([n_sites] * 2, dtype=np.float64)
            for i in range(n_sites - 1):
                h1_effective[i, i+1] = h1_effective[i+1, i] = -1.0  # Hopping t = -1.0
            for i in range(n_sites):
                h1_effective[i, i] += 2 * g_coupling * inner_hh_alpha_values[i].real  # Add Holstein coupling
                print("h1", h1_effective[i, i])
            # Override mf.get_hcore with the new effective h1
            mf.get_hcore = lambda *args: h1_effective
            mf.kernel() # Run electronic SCF

            if not mf.converged:
                print(f"  Warning: Inner Electronic SCF did not converge in inner iteration {inner_iter + 1}.")

            dm = mf.make_rdm1()
            inner_hh_n_i = np.diag(dm)

            # Update alpha values using the self-consistency condition
            inner_hh_alpha_values = -g_coupling / omega0 * inner_hh_n_i

            # Bosonic energy is + omega0 * sum |alpha|^2
            current_inner_hh_bosonic_energy = omega0 * np.sum(np.abs(inner_hh_alpha_values)**2)
            current_inner_hh_total_energy = mf.e_tot + current_inner_hh_bosonic_energy

            inner_hh_alpha_diff = np.linalg.norm(inner_hh_alpha_values - old_inner_hh_alpha_values)
            print(f"  Inner Iter {inner_iter + 1}: E_HH_inner={current_inner_hh_total_energy:>14.8f} Ha, Alpha_diff={inner_hh_alpha_diff:>14.8e}")

            if inner_hh_alpha_diff < inner_hh_alpha_conv_tol:
                print(f"  Inner Hubbard-Holstein SCF Converged in {inner_iter + 1} iterations.")
                break
        else:
            print("  Inner Hubbard-Holstein SCF did NOT converge within max_inner_hh_iter.")

        converged_inner_hh_alpha_values = inner_hh_alpha_values.copy()
        print("alpha values from mean field", converged_inner_hh_alpha_values)

        # Bosonic energy contribution 
        converged_hh_bosonic_energy = omega0 * np.sum(np.abs(converged_inner_hh_alpha_values)**2)
        print("bosonic energy from mean field", converged_hh_bosonic_energy)

        ##fragmentation of the system and BE optimization
        fobj = fragpart(be_type='be2', mol=mol_hub) #fragmentization scheme is BE2
        mybe = BE(mf, fobj)

        # --- Perform BE density matching ---
        start_time_be_opt = perf_counter()
        mybe.optimize(solver='FCI', only_chem=False)
        rdm1_full = mybe.rdm1_fullbasis(return_lo=True, only_rdm1=True, return_RDM2=False)
        stop_time_be_opt = perf_counter()

        print("\nBE RDM1 (full basis):")
        print(rdm1_full)
        print(f'Total runtime for BE optimization in this grand iteration: {stop_time_be_opt - start_time_be_opt} seconds.')

        # --- Update alpha from BE RDM1 ---
        be_n_i = np.diag(rdm1_full)
        print("number of electrons:", np.sum(be_n_i))

        new_alpha_values_for_grand_loop = -g_coupling / omega0 * be_n_i
        print(f"  New alpha values from BE for next Grand Iteration: {new_alpha_values_for_grand_loop.real}")

        be_bosonic_contribution = omega0 * np.sum(np.abs(new_alpha_values_for_grand_loop)**2)
        print("bosonic energy", be_bosonic_contribution)

        be_total_hubbard_holstein_energy = mybe.ebe_tot + be_bosonic_contribution - mf.energy_nuc()
        print(f"BE Total Hubbard-Holstein Energy (Electronic + Bosonic from BE RDM1): {be_total_hubbard_holstein_energy:>14.8f} Ha")

        grand_total_energy_history.append(be_total_hubbard_holstein_energy)

        # --- Check for grand loop convergence ---
        grand_alpha_diff = np.linalg.norm(new_alpha_values_for_grand_loop - old_alpha_values_for_grand_loop)
        print(f"\nGrand Iteration {grand_outer_iter + 1}: Alpha change (norm) from BE: {grand_alpha_diff:>14.8e}")

        alpha_values_for_grand_loop = new_alpha_values_for_grand_loop.copy()

        if grand_alpha_diff < grand_alpha_conv_tol:
            print(f"\n======== Grand SCF Loop Converged in {grand_outer_iter + 1} iterations. ========")
            break

    # --- Final Results After Grand Loop Convergence (per U) ---
    print("\n--- Final Results (after full self-consistency) ---")
    print(f"Final Converged Alpha Values (from BE RDM1): {alpha_values_for_grand_loop.real}")

    # Required per-U final print (U-tagged so it is readable in logs)
    print(
        f"U = {U:>5} | Final Total Hubbard-Holstein Energy (from BE, including bosonic part): "
        f"{grand_total_energy_history[-1]:>14.8f} Ha"
    )

    print("nuclear repulsion energy", mf.energy_nuc())
    print(f"Total runtime for U = {U}: {perf_counter() - start_time_total_U} seconds.")

    final_energy_by_U[U] = grand_total_energy_history[-1]


# Optional: summary after sweep
print("\n" + "=" * 90)
print("SUMMARY: Final Total Hubbard-Holstein Energy (BE, including bosonic part) vs U")
print("=" * 90)
for U in U_LIST:
    E = final_energy_by_U.get(U, None)
    if E is None:
        print(f"U = {U:>5} | (no result)")
    else:
        print(f"U = {U:>5} | E = {E:>14.8f} Ha")
