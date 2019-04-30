# generalizedHartreeFock
Generalized Hartree Fock simulations. Written in Julia.

Tests can be run by simply including test.jl. It tests up to 3-particle Hamiltonians (for exact results) and 1D spin chains (and alerts you to errors if the HF somehow gets lower energy).

hartree.jl contains the standard Hartree-Fock procedure, and mostly exports one symbol, "HF".

generalized-hartree.jl describes the generalized Hartree-Fock procedure. It firts converts the problem to the Majorana basis and then solves for the density matrix.

hubbard.jl can be used to create Hamiltonians for 1D and 2D Hubbard models.
