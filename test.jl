
TestIter=50

function test1(a, model)
	T::Array{Complex{Float64},2} = zeros(1,1)
	T[1,1] = a
	U = [([1,1,1,1],1e-50)]
	e, orb, dens, fockEps = model(T, U, 1, 0,1e-7,TestIter)
end
function test2(a11, a22, a12, model)
	T::Array{Complex{Float64},2} = zeros(2,2)
	T[1,1] = a11
	T[1,2] = T[2,1] = a12
	T[2,2] = a22
	U = [([1,2,1,2],1e-50)]
	e, orb, dens, fockEps = model(T, U, 2, 0,1e-7,TestIter)
end
function testU(a11, a22, a12, u, model)
	T::Array{Complex{Float64},2} = zeros(2,2)
	T[1,1] = a11
	T[1,2] = T[2,1] = a12
	T[2,2] = a22
	U = [([1,2,1,2],u)]
	e, orb, dens, fockEps = model(T, U, 2, 0,1e-7,TestIter)
end
function testU3(w,x,y,z, u1212, u1313, u2323, u1323, model)
	T::Array{Complex{Float64},2} = zeros(3,3)
	T[1,1] = x
	T[1,2] = T[2,1] = w
	T[2,2] = y
	T[3,3] = z
	U = [([1,2,1,2],u1212), ([1,3,1,3],u1313), ([2,3,2,3],u2323), ([1,3,2,3],u1323)]
	e, orb, dens, fockEps = model(T, U, 3, 0,1e-7,TestIter)
end
function testU3A(w1,w2,w3,x,y,z, u1212, u1313, u2323, u1323, u2123)
	T::Array{Complex{Float64},2} = zeros(3,3)
	T[1,1] = x
	T[2,2] = y
	T[3,3] = z
	T[1,2] = T[2,1] = w1
	T[1,3] = T[3,1] = w2
	T[2,3] = T[3,2] = w3
	U = [([1,2,1,2],u1212), ([1,3,1,3],u1313), ([2,3,2,3],u2323), ([1,3,2,3],u1323), ([2,1,2,3],u2123)]
	eG, orb, dens, fockEps = generalizedHF(T, U, 3, 0,1e-7,TestIter)
	e, orb, dens, fockEps = HF(T, U, 3, 0,1e-7,TestIter)
	println(eG)
	println(e)
end
#Maybe a bit excessive. Supposed to be a very general 4-site Hamiltonian.
function testU4A(w1,w2,w3,w4,w5,w6,v,x,y,z, u1212, u1313, u2323, u2424, u3434, u1323, u2123, u1234, u2124, u1324, u2423)
	T::Array{Complex{Float64},2} = zeros(4,4)
	T[1,1] = v
	T[2,2] = x
	T[3,3] = y
	T[4,4] = z
	T[1,2] = T[2,1] = w1
	T[1,3] = T[3,1] = w2
	T[1,4] = T[4,1] = w3
	T[2,3] = T[3,2] = w4
	T[2,4] = T[4,2] = w5
	T[3,4] = T[4,3] = w6
	U = [([1,2,1,2],u1212), ([1,3,1,3],u1313), ([2,3,2,3],u2323), ([2,4,2,4],u2424), 
	([3,4,3,4],u3434), ([1,3,2,3],u1323), ([2,1,2,3],u2123), ([1,2,3,4],u1234),
	([2,1,2,4],u2124), ([1,3,2,4],u1324), ([2,4,2,3],u2423)]
	eG, orb, dens, fockEps = generalizedHF(T, U, 4, 0,1e-7,TestIter)
	e, orb, dens, fockEps = HF(T, U, 4, 0,1e-7,TestIter)
	println(eG)
	println(e)
end
function spin0Ferm1D(n, t, mu, V, model)
	#H = sum_i (t * ai^ * aj + V * ni * nj + mu * ni)
	T::Array{Complex{Float64},2} = zeros(n,n)
	U::Array{Tuple{Array{Int64,1},Float64},1} = []
	
	e0 = 0
	for i = 1:n
		T[i,i] -= mu
		
		if i<n
			T[i,i+1] += t
			T[i+1,i] += t
		
			push!(U, ([i,i+1,i+1,i], V))
		end
	end
	e, orb, dmat, eps = model(T, U, n, e0, 1e-7,TestIter)
end

function ensureEq(v1, v2, tol=1e-4)
	if(abs(v1-v2) > tol)
		error(string(v1," != ",v2))
	end
end
function ensureBound(v1, v2, tol=1e-4)
	if(v1 > v2 + tol)
		error(string(v1," < ",v2))
	end
end

function doSpinChainTests(model)
	# each case is (t,mu,V,E0)
	cases = [
		(0,-2,0,0),
		(0,1,0,-10),
		(0,1,1,-5),
		(0,1,2,-5),
		(0,2,0,-20),
		(0,2,1,-11),
		(0,2,2,-10),
		(1,-2,0,0),
		(1,-2,1,0),
		(1,-2,2,0),
		(1,-1,0,-1.91121448078),
		(0,-2,1,0),
		(1,-1,1,-1.77220265286),
		(1,-1,2,-1.69028233275),
		(1,0,0,-6.02667418333),
		(1,0,1,-5.30390956594),
		(1,0,2,-5.02039616105),
		(1,1,0,-11.9112144808),
		(1,1,1,-10),
		(1,1,2,-9.26418660472),
		(1,2,0,-20),
		(1,2,1,-15.7427381174),
		(0,-2,2,0),
		(1,2,2,-14.2641866047),
		(0,-1,0,0),
		(0,-1,1,0),
		(0,-1,2,0),
		(0,0,0,0),
		(0,0,1,0),
		(0,0,2,0)
	]
	results = []
	for (t, mu, V, reference) = cases
		result = spin0Ferm1D(10, t, mu, V, model)[1]
		push!(results, (t,mu,V,reference,result))
	end
	
	for (t, mu, V, reference, result) = results
		println("Test ",(t,mu,V),": Expected ",round(reference,digits=4),", got ",round(result,digits=4))
		ensureBound(reference, result)
	end
end
function doSmallTests(model, exact)
	cases = [
		(test1,[-10],-10),
		(test1,[+10],0.0),
		(test2,[-3, +4, 0],-3),
		(test2,[+3, +4, 0],0),
		(test2,[-3, -4, 0],-7),
		(test2,[-3, -4, 1],-7),
		(test2,[-3, -4, 100000],-100003.5),
		
		(testU,[1e-10,0,0,1.0],-1.0),
		(testU,[-3,-4,0,1.0],-8.0),
		
		(testU3,[-1.0, 3.0, -1.0,2.0,1.4,1.6,0.7,1.4],-1.23607),
		(testU3,[-1.0, 3.0, -1.0,-1.0,1.4,1.6,0.7,1.4],-3.45054),
		(testU3,[0.7, -1.1, -0.9, 1.6, -1.6, -0.2, 1., -0.6],-1.70711),
		(testU3,[-1.6, -2., -0.2, -0.1, 0.9, -1.1, -0.1, 1.5],-3.1)
	]
	
	results = []
	for (test, args, reference) = cases
		result = test(args..., model)[1]
		push!(results, (test,args,reference,result))
	end
	
	for (test,args,reference,result) = results
		println("Test ",test,": Expected ",round(reference,digits=4),", got ",round(result,digits=4))
		if(exact)
			ensureEq(reference, result)
		else
			ensureBound(reference, result)
		end
	end
end

include("generalized-hartree.jl")
include("hartree.jl")

doSmallTests(generalizedHF, true)
doSmallTests(HF, false)
doSpinChainTests(generalizedHF)
doSpinChainTests(HF)

println("All Tests Passed")