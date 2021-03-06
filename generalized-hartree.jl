using SparseArrays
using LinearAlgebra

###
# The routine runs a Hartree-Fock calculation.
# It's taking as "input" the background Hamiltonian (`hCore`),
# a (not necessarily orthogonal) set of orbitals with known overlap integrals (`overlap`),
# and a sparse 4-tensor of electron-electron repulsion terms (`eeRep`). It also of course
# uses `n`, the number of orbitals, and `nOcc`, the number of occupied orbitals.
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


#Takes a k-tensor with entries of the form ([i,j,...,k,l],val),
#with possibly redundant values, then "collapses" values together.
function collapseSparseTensor(tensor::T) where T
	if tensor == [] return [] end
	
	rank = tensor[1][1]
	sortedRows = sort(tensor, by=first)
	
	# We'll aggregate the new rows here
	# Start with just the first row
	collapsed::T = [sortedRows[1]]
	
	for tup in sortedRows[2:end]
		if(collapsed[end][1] == tup[1])
			#Add this row into the previous one
			collapsed[end] = (tup[1],collapsed[end][2]+tup[2])
		else
			#Append the new row
			push!(collapsed, tup)
		end
	end
	return filter(x -> x[2] != 0, collapsed)
end

function scaleSparseTensor(tensor, scalar)
	res = map(x -> (x[1], x[2]*scalar), tensor)
	return res
end

function conjSparseTensor(tensor)
	res = map(x -> (x[1], conj(x[2])), tensor)
	return res
end

function addSparseTensors(tensor1, tensor2)
	if length(tensor1) == 0
		return tensor2
	end
	if length(tensor2) == 0
		return tensor1
	end
	
	if(length(tensor1[1][1]) != length(tensor2[1][1]))
		error(string("Tensors don't match in rank: ",length(tensor1[1][1])," != ",length(tensor2[1][1])))
	end
	return collapseSparseTensor([tensor1; tensor2])
end

#antisymmetrizes U in the fermion basis
function antisymmetrizeUFerm(tensor::Array{Tuple{Array{Int64,1},T},1}) where T<:Number
	rank = length(tensor[1][1])
	if(rank != 4)
		error(string("U should be rank 4, found rank ",rank))
	end
	
	#add Uijkl + Uklij*
	tensor = addSparseTensors(tensor, conjSparseTensor(tranposeSparseTensor(tensor, [3,4,1,2])))
	#add Uijkl - Ujikl
	tensor = addSparseTensors(tensor, scaleSparseTensor(tranposeSparseTensor(tensor, [2,1,3,4]), -1))
	#add Uijkl - Uijlk
	tensor = addSparseTensors(tensor, scaleSparseTensor(tranposeSparseTensor(tensor, [1,2,4,3]), -1))
	#scale back down by the 8 copies combined
	tensor = scaleSparseTensor(tensor, 1.0/8.0)
	return tensor
end

#Take a U tensor and builds the corresponding u tensor
#returns (U, T, E0), where T and E0 are additional terms that arise
function fermToMajSparse(tensor, n)
	res = []
	resT = zeros(Complex{Float64},2n,2n)
	resE = 0
	
	if length(tensor) == 0
		return (tensor, Array{Float64}(resT), resE)
	end
	
	rank = length(tensor[1][1])
	if(rank != 4)
		error(string("U should be rank 4, found rank ",rank))
	end
	
	for (fermInd, val) = tensor
		#2 parts for each operator
		#4 operators -> 16 product terms
		
		#loop over the 16 terms
		for way4 = 0:15
			#pull out bits for each
			bits = map(x -> sign(way4 & x), [1,2,4,8])
			#get Majorana operator numbers
			inds = map(x -> 2*x[1] + x[2] - 1, zip(fermInd, bits))
			#compute the coefficient
			scalar = (-im)^bits[1] * (-im)^bits[2] * (+im)^bits[3] * (+im)^bits[4] * val
			
			#check for repeated operators
			if(inds[1] == inds[3])
				scalar *= -1
				if(inds[2] == inds[4])
					resE += scalar
				else
					resT[inds[2],inds[4]] += scalar
				end
			elseif (inds[1] == inds[4])
				scalar *= +1
				if(inds[2] == inds[3])
					resE += scalar
				else
					resT[inds[2],inds[3]] += scalar
				end
			elseif (inds[2] == inds[3])
				scalar *= +1
				resT[inds[1],inds[4]] += scalar
			elseif (inds[2] == inds[4])
				scalar *= -1
				resT[inds[1],inds[3]] += scalar
			else
				#it actually is a 4-fermion term, put it in U
				#put this in the tensor
				push!(res, (inds, scalar))
			end
		end
	end
	res = collapseSparseTensor(res)
	
	#now antisymmetrize U
	
	#add Uijkl + Uiklj + Uiljk
	res = addSparseTensors(addSparseTensors(res, tranposeSparseTensor(res, [1,3,4,2])), tranposeSparseTensor(res, [1,4,2,3]))
	#add Uijkl + Uklij
	res = addSparseTensors(res, tranposeSparseTensor(res, [3,4,1,2]))
	#add Uijkl + Ujilk
	res = addSparseTensors(res, tranposeSparseTensor(res, [2,1,4,3]))
	#add Uijkl - Ujikl
	res = addSparseTensors(res, scaleSparseTensor(tranposeSparseTensor(res, [2,1,3,4]), -1))
	#scale back down by the 24 copies combined
	res = scaleSparseTensor(res, 1.0/24)
	res = collapseSparseTensor(res)
	
	#and antisymmetrize resT
	resT = (resT - transpose(resT))/2
	#put T in purely real form by dividing by i	
	resT = Array{Float64,2}(resT/im)
	
	#assert energy shift is purely real
	resE = Float64(resE)
	
	return (res, resT/8, resE/16)
end

#Take a T tensor and builds the corresponding t tensor.
#Might contribute a constant part, returned as e0.
function fermToMajDense(mat::Array{Complex{Float64},2})
	n = size(mat)[1]
	res = zeros(2n,2n)
	e0 = 0
	for i = 1:n, j = 1:n
		re_ = real(mat[i,j])
		im_ = imag(mat[i,j])
		res[2i-1,2j] = +re_/2
		res[2i,2j-1] = -re_/2
		res[2i,2j] = -im_/2
		res[2i-1,2j-1] = -im_/2
		if i==j
			e0 += re_/2
		end
	end
	#antisymmetrize
	res = (res - transpose(res))/2
	return (e0, res)
end

#like fermToMajDense but for number-violating terms
function NVfermToMajDense(mat::Array{Complex{Float64},2})
	n = size(mat)[1]
	res = zeros(2n,2n)
	e0 = 0
	for i = 1:n, j = 1:n
		re_ = real(mat[i,j])
		im_ = imag(mat[i,j])
		res[2i-1,2j-1] = -re_/2
		res[2i,2j] = -re_/2
		res[2i,2j] = +im_/2
		res[2i-1,2j-1] = -im_/2
		if i==j
			e0 += re_/2
		end
	end
	#antisymmetrize
	res = (res - transpose(res))/2
	return (e0, res)
end

#Take a covariance matrix, and measure how much it differs from a number
#eigenstate. If this number is significantly larger than zero, the gHF is
#breaking number symmetry.
function checkNumberEigenstate(mat::Array{Float64,2})
	n = Int64(size(mat)[1]/2)
	tot = 0
	for i = 1:n, j = 1:n
		tot += abs(mat[2i-1,2j] + mat[2i,2j-1])
		tot += abs(mat[2i,2j] - mat[2i-1,2j-1])
	end
	return tot
end

#Given a tensor U[i,j,k,l], maps it to entries
#of the form U[i + j*n, k + l*n].
function flatten4Tensor(tensor, n)
	if length(tensor) == 0
		return sparse([],[],[],n^2,n^2)
	end
	
	rank = length(tensor[1][1])
	if(rank != 4)
		error(string("Can only flatten tensor of rank 4, not rank",rank))
	end
	Is = map(x -> (x[1][1] + (x[1][2]-1)*n), tensor)
	Js = map(x -> (x[1][3] + (x[1][4]-1)*n), tensor)
	Vs = map(x -> x[2], tensor)
	return sparse(Is, Js, Vs, n^2, n^2)
end

function tranposeSparseTensor(tensor, perm)
	if length(tensor) == 0
		return tensor
	end
	
	rank = length(tensor[1][1])
	if(rank != length(perm))
		error(string("Cannot transpose rank ",rank," with permutation ",perm))
	end
	
	changeEntry = entry -> (map(i -> entry[1][perm[i]], 1:rank), entry[2])
	return map(changeEntry, tensor)
end

#Assumes the orbitals are already orthonormal.
#numViolate is for number-violating terms. (c_i* c_j* + c_i c_j)*numViolate[i,j]. Not implemented yet...
#oda=true enabled an Optimal Damping Algorithm
#fockAvg=true will linearly combine fock matrices from previous iterations to stabilize search. Does not work well with oda=true.
function generalizedHF(tMat, uEntries, n, enuc=0; tol=1e-7, maxiter=20,
	numViolate=(), oda=true, fockAvg=false, tMAThw0=0)
	#playing with these might help stability.
	tMAThw = tMAThw0
	RandInit = 0.8
	# amount to average with previous iteration's density matrix.
	OLDFOCKw = 0
	
	#Change basis from ak,ak* into c2k,c2k-1.
	e0T, tMatC = fermToMajDense(tMat)
	enuc += e0T
	
	uMat = uEntries
	uMat = antisymmetrizeUFerm(uMat)
	uMat, t0U, e0U = fermToMajSparse(uMat, n)
	uMatC::SparseMatrixCSC{Float64,Int64} = flatten4Tensor(uMat, 2*n)
	uMatC *= 1/8
	
	tMatC += t0U
	enuc += e0U
	
	perturb = RandInit * (rand(2*n,2*n).-0.5) #randomly perturb initial Hamiltonian to break symmetry if necessary
	fock0 = tMatC + perturb - transpose(perturb)
	
	#Begin the main loop
	iter=1
	oldfock_raise_iter = 0
	smoothedIter=0
	eOld = e0 = 0
	densityMat::Array{Float64,2} = zeros(2*n,2*n)
	
	dFock = 0
	println("Start loop")
	
	
	while true
		if VERBOSE
			println()
			println("Fock0: ",fock0)
		end
		
		fockEigen = eigen(Hermitian(Array(1im * fock0)))
		# Eigen already sorts eigenvalues, when given a symmetric matrix.
		fockEps = fockEigen.values / 1im
		# The eigenfunctions, in the orthonormal basis.
		fockOrbitals = fockEigen.vectors
		
		# Instead of choosing signs using the fock energies, we can use the Hamiltonian energy.
		# In theory setting tMAThw = 1 would do this, and should improve stability. In practice, it hurts.
		# Not sure why. Leaving it at 0 would do nothing, and so this code is skipped completely.
		if tMAThw != 0
			h = (tMAThw*tMatC + 1*fock0)
			fockEps = imag(diag(conj(transpose(fockOrbitals)) * h * fockOrbitals))*im
		end
		
		#If we're doing Optimal Damping Algorithm, save the old density matrix
		if iter > 1 && oda
			oldDmat = densityMat
		end
		
		function doGradDescent()
			println("STARTING ENERGY: ",e0+enuc)
			
			#Compute gradient
			Agrad = (densityMat * fock0 - fock0 * densityMat)
			#Scale down gradient to a 'normalized' scale
			Agrad /= maximum(abs.(Agrad)) * max(n, 5) * 2
			println(maximum(abs.(Agrad)))
			Asq = Agrad*Agrad
			A4 = Asq*Asq
			#accurate to 1e-8 for x in [-0.2,0.2]. Definitely good enough for this
			expA = I + Agrad + Asq/2 + Agrad*Asq/6 + A4/24 + Agrad*A4/120
				
			while true
				#If uncommented, these will tell you how far the density matrix is from eigenvalues just in [-I ... I].
				#println("starting eval error: ",maximum(abs.(eigen(Hermitian(Array(1im * densityMat))).values/1im).-1))
				trialDensityMat = expA * densityMat * transpose(expA)
				#println("rotated eval error: ",maximum(abs.(eigen(Hermitian(Array(1im * densityMat))).values/1im).-1))
				
				fock0 = copy(tMatC)
				uContrib = reshape(uMatC * reshape(trialDensityMat, (4*n*n)), (2*n,2*n))
				fock0 += 6 * uContrib
				
				e1 = tr(densityMat * (tMatC + fock0))/4
				
				if e1 < e0
					#good, we improved, let's try again with a double step size
					densityMat = trialDensityMat
					expA ^= 2
					e0 = e1
					println("NEW ENERGY: ",e1+enuc)
				else
					#Rotated too far, let's call it done
					break
				end
			end
		end
		
		# Build the density matrix by choosing signs of +-im in the Fock matrix
		densityMat = real(fockOrbitals * spdiagm(0 => map(sign, fockEps)) * conj(transpose(fockOrbitals)))
		if VERBOSE
			println("Density: ",densityMat)
		end
		
		
		# Build the new Fock matrix. First, back up the old one.
		oldFock = copy(fock0)
		
		# The new fock matrix starts with the T term
		fock0 = copy(tMatC)
		# Then we multiply the U term with the current density, and add that in
		uContrib = reshape(uMatC * reshape(densityMat, (4*n*n)), (2*n,2*n))
		#println("U contrib: ",uContrib)
		fock0 += 6 * uContrib
		#We've now built a new fock matrix and can repeat the loop.
		
		if iter > 1 && oda
			#We know the energy at oldFock is e0
			#Compute new energy at fock0:
			e1 = tr(densityMat * (tMatC + fock0))/4
			#Compute the energy at (oldFock + fock0)/2. The fock matrix here is the average of the two
			#others, and the density matrix is also the average.
			eHalf = tr(((oldDmat+densityMat)/2) * (tMatC + (oldFock+fock0)/2))/4
			#Given energies e0, e1, and e0.5, we can compute parameters in ax^2+bx+c.
			a = 2*(e0 - 2*eHalf + e1)
			b = -3*e0 + 4*eHalf - e1
			#Parabola vertex is at -b/2a
			vert = -b/(2a)
			println("ODA vertex at ",vert)
			#Curve may be convex down, in which case take the lower endpoint
			if(eHalf > (e0+e1)/2)
				#convex down
				println("Convex down")
				vert = e0 > e1 ? 1 : 0
			end
			if(vert >= 1)
				#vertex is outside our region of density matrices, just keep what we computed earlier
			elseif(vert <= 0)
				#we chose a strictly bad direction? This shouldn't happen often. If it is, we're choosing
				#a bad direction for our new density matrix.
				println("Warning: parabola vertex <= 0. ODA cannot run. Falling back to gradient descent.")
				println("Energies: ",e0+enuc,eHalf+enuc,e1+enuc)
				#put it back where it was before
				densityMat = oldDmat
				fock0 = oldFock
				#Fall back to gradient descent
				doGradDescent()
			else
				#Use the interpolation parameter
				densityMat*= vert
				fock0 *= vert
				densityMat += oldDmat * (1-vert)
				fock0 += oldFock * (1-vert)
				
				println("ODA Energy: ",enuc+tr(densityMat * (tMatC + fock0))/4)
				
				#These lines will do a mapping (x -> (3x - x^3)/2). If working with only pure states, this will help
				#make sure they stay pure after rotation: it maps eigenvalues in the vicinty of I to I, and in the vicinity of -I to -I.
				
				#The idea is to take impure states and 'extremize' them in the (convex) space of allowed matrices.
				for purifications = 1:3
					oldDmat = densityMat
					densityMat = (3*densityMat + densityMat*densityMat*densityMat)/2
					impurity = sum(abs.(densityMat - oldDmat)) / n
					#println("Removed impurity of ",impurity)
					
					impurity < 0.01 && break
				end
				
				#Compute new energy
				fock0 = copy(tMatC)
				uContrib = reshape(uMatC * reshape(densityMat, (4*n*n)), (2*n,2*n))
				fock0 += 6 * uContrib
				
				println("Purified ODA Energy: ",enuc+tr(densityMat * (tMatC + fock0))/4)
			end
		end
		
		# before we do, compute some numbers on how it's converging
		dFockOld = dFock
		dFock = oldFock - fock0
		dFockE = tr(densityMat * dFock)
		
		# Compute new energy
		# In the original basis, the energy computation would be:
		#   eNew = tr(densityMat * (tMatC + fock0))/2
		# In the Majorana basis, we have twice as many "modes", so
		# we need to divide by 2 an extra time.
		eNew = tr(densityMat * (tMatC + fock0))/4
		eTot = eNew + enuc
		println("Iter #",iter,",  E = ",eTot,",  dFock = ",sum(abs.(dFock)))
		
		if(fockAvg)
			if(iter >= oldfock_raise_iter + 2 && (sum(abs.(dFock)) > 0.9 * sum(abs.(dFockOld))))
				OLDFOCKw += 1/(2+3*OLDFOCKw)
				oldfock_raise_iter = iter
				println("fockAvg weight = ",OLDFOCKw)
			end
			fock0 = (fock0 + OLDFOCKw*oldFock)/(1+OLDFOCKw)
		end
		
		if iter > 1
			eOld = e0
		end
		
		e0 = eNew
		
		if(e0 > eOld)
			println(e0+enuc," > ",eOld+enuc)
			println()
			#error(string(e0," > ",eOld))
		end
		
		#Termination condition: energy stabilized?
		if ((abs(e0 - eOld) < tol) && (abs(dFockE) < tol) && iter > 2) || (iter > maxiter)
			println()
			println("Energy variation at current state, from varying Fock matrices: ",dFockE)
			println("Energy variation between iterations: ",abs(e0 - eOld))
			
			nAvg = sum(map(i -> (-densityMat[2*i-1,2*i]+1)/2, 1:n))
			println("Expected particles: ",nAvg)
			numberState = checkNumberEigenstate(densityMat)
			println("Distance from number eigenstate: ",numberState)
			return (eTot, fockOrbitals, densityMat, fockEps)
		end
		#Otherwise, loop.
		iter+=1
		
	end
end

#Assumes the orbitals are already orthonormal.
function RandomHF(tMat, uEntries, n, enuc=0; tol=1e-7, maxiter=1000)
	#Change basis from ak,ak* into c2k,c2k-1.
	e0T, tMatC = fermToMajDense(tMat)
	enuc += e0T
	
	uMat = uEntries
	uMat = antisymmetrizeUFerm(uMat)
	uMat, t0U, e0U = fermToMajSparse(uMat, n)
	uMatC::SparseMatrixCSC{Float64,Int64} = flatten4Tensor(uMat, 2*n)
	uMatC *= 1/8
	
	tMatC += t0U
	enuc += e0U
	
	#Begin the main loop
	iter=1
	iterSinceLastBetter = 0
	eBest = 0
	dBest = 0
	
	println("Start loop")
	
	while true
		
		dMat = rand(2*n,2*n)
		dMat = dMat - transpose(dMat)
		if iter > 1
			dMat = 2 * dMat * (0.98^iterSinceLastBetter) + dBest
		end
		dMatEigen = eigen(Hermitian(Array(1im * dMat)))
		densityMat::Array{Float64,2} = real(dMatEigen.vectors * spdiagm(0 => map(sign, dMatEigen.values/1im)) * conj(transpose(dMatEigen.vectors)))
		
		# The new fock matrix starts with the T term
		fock0 = copy(tMatC)
		# Then we multiply the U term with the current density, and add that in
		uContrib = reshape(uMatC * reshape(densityMat, (4*n*n)), (2*n,2*n))
		#println("U contrib: ",uContrib)
		fock0 += 6 * uContrib
		#We've now built a new fock matrix and can repeat the loop.
		
		# Compute new energy
		# In the original basis, the energy computation would be:
		#   eNew = tr(densityMat * (tMatC + fock0))/2
		# In the Majorana basis, we have twice as many "modes", so
		# we need to divide by 2 an extra time.
		eNew = tr(densityMat * (tMatC + fock0))/4
		eTot = eNew + enuc
		#println("Iter #",iter,",  E = ",eTot,",  dFock = ",sum(abs.(dFock)))
		
		if iter == 1 || eTot < eBest
			eBest = eTot
			dBest = densityMat
			iterSinceLastBetter = 0
		else
			iterSinceLastBetter += 1
		end
		
		#Termination condition: energy stabilized?
		if (iter > maxiter)
			println()
			
			nAvg = sum(map(i -> (-dBest[2*i-1,2*i]+1)/2, 1:n))
			println("Expected particles: ",nAvg)
			println("Best energy: ",eBest)
			return (eBest, 0*dBest, dBest, 0*dBest) #no actual fock matrices
		end
		#Otherwise, loop.
		iter+=1
		
	end
end

VERBOSE = false