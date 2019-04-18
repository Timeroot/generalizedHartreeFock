using SparseArrays
using LinearAlgebra

###
# The routine runs a Hartree-Fock calculation.
# It's taking as "input" the background Hamiltonian (`hCore`),
# a (not necessarily orthogonal) set of orbitals with known overlap integrals (`overlap`),
# and a sparse 4-tensor of electron-electron repulsion terms (`eeRep`). It also of course
# uses `n`, the number of orbitals, and `nOcc`, the number of occupied orbitals.
#
# Passing -1 for nOcc will allow a variable occupation number, i.e. Grand canonical
#
# Optional arguments start with the number of electrons per orbital ('orbitalsize') --
# for Restricted Hartree-Fock this should be 2, but for generalized Hartree-Fock this
# should be 1. Next is a constant offset in energy (`enuc`) that represents
# non-electron terms in the Hamiltonian, and a tolerance (`tol`) at which the energy
# needs to stabilize. The `spinSymm` parameter will enforce that the Fock matrix is
# symmetric under spin inversion, to fix spin contamination issues.
#
# It returns (energy, orbitals, densityMat, energy_levels). `orbitals` are sorted by energy, and
# include all orbitals, not just the occupied ones. For instance, `orbitals[:,1]` is
# the lowest-energy orbital. `energy_levels` is the computed Fock energies for all orbitals.
###

function hartreefock(hCore, overlap, eeRep, n, nOcc, orbitalsize=1, enuc=0, tol=1e-7, maxiter=20)
	
	# Use the overlap integrals to build an orthogonalizing symmetric matrix S.
	# This involves solving an eigensystem on the overlap integrals, once.
	#
	# If overlap is already the identity, then the input was already orthonormal,
	# and we skip this step (and set orthoS to an identity matrix.)
	#
	if overlap == I
		orthoS = I
	else
		overlapEigen = eigen(Array(overlap))
		orthoS = Symmetric(overlapEigen.vectors * Diagonal(overlapEigen.values.^(-0.5)) * transpose(overlapEigen.vectors))
	end
	
	eeRep = orderUFerm(eeRep)
	
	fock0 = hCore + 10.0 * Tridiagonal(rand(n,n).-0.5) #randomly perturb initial Hamiltonian to break symmetry if necessary
	eOld = e0 = tr(hCore + fock0)*(orbitalsize/2)
	
	#Begin the main loop
	iter=1
	smoothedIter=0
	
	dFock = 0
	
	while true
		
		fockEigen = eigen(Hermitian(Array(orthoS * fock0 * orthoS)))
		# Eigen already sorts eigenvalues, when given a symmetric matrix.
		fockEps = fockEigen.values
		# The eigenfunctions, in the orthonormal basis.
		fockOrbitalsOrtho = fockEigen.vectors
		# The eigenfunctions, back in the orbital basis.
		fockOrbitals = orthoS * fockOrbitalsOrtho
		
		# If nOcc is nonnegative, occupy that many orbitals.
		# If it's -1, fill only up to 0 energy.
		effectiveNOcc = (nOcc == -1) ? sum(map(x->0<-x, fockEps)) : nOcc
		densityMat = fockOrbitals[:,1:effectiveNOcc] * transpose(fockOrbitals[:,1:effectiveNOcc])
		
		
		dFockE = tr(densityMat * dFock)
		
		# Build the new Fock matrix. Awkwardly involves walking through the eeRep sparse 4-tensor, so
		# this might be significant for performance, depending on how sparse that tensor is.
		oldFock = 1*fock0
		
		fock0 = 1*hCore
		
		for eeRepTerm in eeRep
			i,j,k,l, v = eeRepTerm
			
			# We need to do each addition a few different ways, to respect
			# the symmetry. So we make one function for the basic operation.
			
			# We assume that i<=j and k<=l. This means we can check for (1,4,4,1) symmetry
			# by just asking if (i==k)&&(j==l), because it will always be in the (1,4,1,4) form.
			
			function sym0(i,j,k,l)
				val = densityMat[k,l] * v * 0.5
				fock0[i,j] += val
				fock0[j,i] += val
				
				fock0[i,k] += -densityMat[j,l] * v * 0.5
				fock0[j,k] += -densityMat[i,l] * v * 0.5
			end
			function sym1(i,j,k,l)
				sym0(i,j,k,l)
				sym0(i,j,l,k)
			end
			function sym2(i,j,k,l)
				sym1(i,j,k,l)
				sym1(k,l,i,j)
			end
			sym2(i,j,k,l)
		end
		#We've now build a new fock matrix and can repeat.
		
		dFock = oldFock - fock0
		
		# Compute new energy
		eOld = e0
		e0 = Float64(tr(densityMat * (hCore + fock0))*(orbitalsize/2))
		eTot = e0 + enuc
		
		#Stabilize fock matrix
		fock0 = (fock0 + OLDFOCKw*oldFock)/(1+OLDFOCKw)
		
		println("Iter #",iter," E = ",eTot)
		if(n>7 && VERBOSE)
			println("Densities start: ",diag(densityMat)[1:7],"...")
		end
		
		#Termination condition: energy stabilized?
		if ((abs(e0 - eOld) < tol) && (abs(dFockE) < tol)) || (iter > maxiter)
			println("Energy variation at current state, from varying Fock matrices: ",dFockE)
			println("Energy variation between iterations: ",abs(e0 - eOld))
			return (eTot, fockOrbitals, densityMat, fockEps)
		end
		
		#Otherwise, loop.
		iter+=1
	end
end

#puts U in the canonical order (i,j,k,l) where i<=j and k<=l.
function orderUFerm(uFerm)
	return map(x -> (min(x[1],x[2]),max(x[1],x[2]),min(x[3],x[4]),max(x[3],x[4]), x[5]*sign(x[2]-x[1])*sign(x[4]-x[3])), uFerm)
end

#Does normal hartree fock, in the grand canonical ensemble.
#Mostly just a wrapper to take the same arguments are generalizedHF.
function HF(tMat, uEntries, n, enuc=0, tol=1e-7, maxiter=20)
	eeRep = map(x -> (x[1][1],x[1][2],x[1][3],x[1][4], x[2]), uEntries)
	return hartreefock(tMat, I, eeRep, n, -1, 1, enuc, tol, maxiter)
end

#Does normal hartree fock, at half filling fixed N.
#Mostly just a wrapper to take the same arguments are generalizedHF.
function halfFillingHF(tMat, uEntries, n, enuc=0, tol=1e-7, maxiter=20)
	eeRep = map(x -> (x[1][1],x[1][2],x[1][3],x[1][4], x[2]), uEntries)
	return hartreefock(tMat, I, eeRep, n, Int64(n/2), 1, enuc, tol, maxiter)
end

function kFillingHF(nOcc)
	function x(tMat, uEntries, n, enuc=0, tol=1e-7, maxiter=20)
		eeRep = map(x -> (x[1][1],x[1][2],x[1][3],x[1][4], x[2]), uEntries)
		return hartreefock(tMat, I, eeRep, n, nOcc, 1, enuc, tol, maxiter)
	end
	return x
end