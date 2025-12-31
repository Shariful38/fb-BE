# fb-BE
**fb-BE** is an extension of the QuEmb framework for Bootstrap Embedding (BE), designed to facilitate the study of fermion–boson lattice Hamiltonians, with particular emphasis on the Hubbard–Holstein model. This repository extends **QuEmb** by incorporating supplementary features for handling bosonic degrees of freedom through a coherent-state formulation.

The original **QuEmb** codebase offers the fundamental embedding framework, encompassing fragment construction, bath orbital generation derived from the mean-field one-particle reduced density matrix (1-RDM), and the enforcement of global reduced density matrix (RDM) consistency. **fb-BE** incorporates fermion–boson physics within this infrastructure.

## Features

* **Mixed Fermi-Boson system:**  Integrates electronic correlation (through Bootstrap Embedding) with bosonic phonon modes represented in a coherent basis, employing a unified self-consistent loop to accurately capture the electron-phonon coupling effect in strongly correlated materials.
* **High Accuracy:** Benchmarks ground state energies against exact DMRG.
* **Flexible Parameters:** Enables straightforward modification of system parameters including size (n_sites), number of electrons, Hubbard interaction (U), phonon frequency (omega), and coupling strength (g).

## Installation

### Prerequisites
* Python 3.6+
* [PySCF](https://pyscf.org/)
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* **[QuEmb](https://github.com/oimeitei/quemb)** (Required for BE calculations)


