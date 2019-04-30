using SparseArrays
using LinearAlgebra
using DelimitedFiles

include("hartree.jl")
include("generalized-hartree.jl")

println("Libraries loaded.")

#1D Hubbard model
# Arguments: 
# * model (HF or generalizedHF. Or, nothing, to just get the parameters for later.)
# * number of sites
# * Hopping term
# * Onsite interaction
# * Chemical potential
# * Periodic boundary conditions (true or false)
# * Number of tries (take the minimum energy)
# * Any further named options to be passed to the model
function hubbard1D(model, nSites; hoppingT=1, U=0, mu=0, periodic=false, restarts=1, options=())
	# compute number of orbitals, 2 spins per site
	n = 2*nSites

	T::Array{Complex{Float64},2} = zeros(n,n)
	Umat::Array{Tuple{Array{Int64,1},Float64},1} = []
	e0 = 0
	
	# add hopping term, T
	# step through each site, add hopping two site+2 (next site, same spin)
	for i = 1:n
		if (i+1<n) || periodic
			i2 = (i+1<n) ? i+2 : i+2-n
			T[i,i2] -= hoppingT
			T[i2,i] -= hoppingT
		end
	end
	
	# add onsite terms, U
	# step two orbitals at a time (one site)
	for i = 1:2:n
		i2 = i+1
		
		# U(ci*ci-1/2)(cj*cj-1/2) = U ci*cj*cjci + (-U/2)ci*ci + (-U/2)cj*cj + U/4
		push!(Umat, ([i,i2,i2,i], U))
		T[i,i] -= U/2
		T[i2,i2] -= U/2
		e0 += U/4
	end
	
	#add chemical potential
	for i = 1:n
		T[i,i] -= mu
	end
	
	if model == nothing
		#Don't run a simulation, just spit out the Hamiltonian.
		return (T,Umat,n,e0)
	end
	
	#This could be a one-liner by using map+argmin, but that would save all density matrices in memory,
	#which could potentially be a pain. This discards them after each try.
	bestResults = model(T, Umat, n, e0; options...)
	for retry = 2:restarts
		newResults = model(T, Umat, n, e0; options...)
		if(newResults[1] < bestResults[1])
			bestResults = newResults
		end
	end
	return bestResults
end

function hubbard2D(model, nSize; hoppingT=1, U=0, mu=0, periodic=false, restarts=1, options=())
	# compute number of orbitals, 2 spins per site
	n = 2*nSize*nSize

	T::Array{Complex{Float64},2} = zeros(n,n)
	Umat::Array{Tuple{Array{Int64,1},Float64},1} = []
	e0 = 0
	
	#convert between (x,y,spin) coordinates and the i indices used to index orbitals
	iToXYS = i -> (Int64(floor((i-1)/(2*nSize))),Int64(floor((i-1)/2))%nSize,(i-1)%2)
	xysToI = xys -> xys[1]*2*nSize + xys[2]*2 + xys[3] + 1 
	
	# add hopping term, T
	# step through each site
	for i = 1:n
		x,y,s = iToXYS(i)
		
		#add +1 in the x direction hopping
		if (x<n-1) || periodic
			x2 = (x+1)%nSize
			i2 = xysToI((x2,y,s))
			T[i,i2] -= hoppingT
			T[i2,i] -= hoppingT
		end
		#and then +1 in y
		if (y<n-1) || periodic
			y2 = (y+1)%nSize
			i2 = xysToI((x,y2,s))
			T[i,i2] -= hoppingT
			T[i2,i] -= hoppingT
		end
	end
	if(nSize == 2) T = T/2 end
	
	# add onsite terms, U
	# step two orbitals at a time (one site)
	for i = 1:2:n
		i2 = i+1
		
		# U(ci*ci-1/2)(cj*cj-1/2) = U ci*cj*cjci + (-U/2)ci*ci + (-U/2)cj*cj + U/4
		push!(Umat, ([i,i2,i2,i], U))
		T[i,i] -= U/2
		T[i2,i2] -= U/2
		e0 += U/4
	end
	
	#add chemical potential
	for i = 1:n
		T[i,i] -= mu
	end
	
	if model == nothing
		#Don't run a simulation, just spit out the Hamiltonian.
		return (T,Umat,n,e0)
	end
	
	#This could be a one-liner by using map+argmin, but that would save all density matrices in memory,
	#which could potentially be a pain. This discards them after each try.
	bestResults = model(T, Umat, n, e0; options...)
	for retry = 2:restarts
		newResults = model(T, Umat, n, e0; options...)
		if(newResults[1] < bestResults[1])
			bestResults = newResults
		end
	end
	return bestResults
end

function writeData1D()
    xs = -20:0.5:8
	
	HFdataN2 = map(u -> hubbard1D(HF,25,1,u,-2,true,7), xs)
	gHFdataN2 = map(u -> hubbard1D(generalizedHF,25,1,u,-2,true,3), xs)

	HFdata0 = map(u -> hubbard1D(HF,25,1,u,0,true,7), xs)
	gHFdata0 = map(u -> hubbard1D(generalizedHF,25,1,u,0,true,3), xs)

	HFdata1 = map(u -> hubbard1D(HF,25,1,u,1,true,7), xs)
	gHFdata1 = map(u -> hubbard1D(generalizedHF,25,1,u,1,true,3), xs)

	HFdata2 = map(u -> hubbard1D(HF,25,1,u,2,true,5), xs)
	gHFdata2 = map(u -> hubbard1D(generalizedHF,25,1,u,2,true,3), xs)

	HFdata4 = map(u -> hubbard1D(HF,25,1,u,4,true,5), xs)
	gHFdata4 = map(u -> hubbard1D(generalizedHF,25,1,u,4,true,3), xs)

	data = [xs HFdataN2 HFdata0 HFdata1 HFdata2 HFdata4 gHFdataN2 gHFdata0 gHFdata1 gHFdata2 gHFdata4]
	writedlm("D:\\Timeroot\\Documents\\UCSB_G1\\GMPS\\hubbard1D.csv", data, ",")
end

function writeData1DMu05()
    xs = -6:0.2:6
	
	# HFdata05 = map(u -> hubbard1D(HF,4,1,u,0.5,false,14), xs)
	# gHFdata05 = map(u -> hubbard1D(generalizedHF,4,1,u,0.5,false,20), xs)
	# data = [xs HFdata05 gHFdata05]
	# writedlm("D:\\Timeroot\\Documents\\UCSB_G1\\GMPS\\hubbard1Dmu05.csv", data, ",")
	
	
	# HFdata05 = map(u -> hubbard1D(HF,4,1,u,-0.5,false,14), xs)
	# gHFdata05 = map(u -> hubbard1D(generalizedHF,4,1,u,-0.5,false,20), xs)
	# data = [xs HFdata05 gHFdata05]
	# writedlm("D:\\Timeroot\\Documents\\UCSB_G1\\GMPS\\hubbard1DmuN05.csv", data, ",")
	
	
	HFdata05 = map(u -> hubbard1D(HF,6,1,u,0.5,false,14), xs)
	gHFdata05 = map(u -> hubbard1D(generalizedHF,6,1,u,0.5,false,20), xs)
	data = [xs HFdata05 gHFdata05]
	writedlm("D:\\Timeroot\\Documents\\UCSB_G1\\GMPS\\hubbard1Dmu05_6.csv", data, ",")
	
	
	#HFdata05 = map(u -> hubbard1D(HF,20,1,u,0.5,false,14), xs)
	#gHFdata05 = map(u -> hubbard1D(generalizedHF,20,1,u,0.5,false,20), xs)
	#data = [xs HFdata05 gHFdata05]
	#writedlm("D:\\Timeroot\\Documents\\UCSB_G1\\GMPS\\hubbard1Dmu05_20.csv", data, ",")
end