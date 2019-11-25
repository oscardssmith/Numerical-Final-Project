using LinearAlgebra
using BenchmarkTools
using Random
include("matMult.jl")

function getTimes(range,T,numTests,multfunction)
    times=zeros(Float64,size(range,1))
    for i in 1:size(range,1)
        N=range[i]
        vals=zeros(Float64,numTests)
        for tests in 1:numTests

            A=rand(T,N,N)
            B=rand(T,N,N)
            #println(tests)
            vals[tests]=@elapsed multfunction(A,B)
        end
        times[i]=minimum(vals)
    end
    return times
end

function testMults(N::Int64,T)

    slowValid=0
    stupidValid=0
    strassenValid=0
    A = rand(T,N,N)
    B = rand(T,N,N)

    #precompile
    x = rand(T,16,16)
    mult(x,x)
    naiveMult(x,x)
    blockedMult(x,x)
    #@code_warntype blockedMult!(x,x,x)
    strassenNoRecurse(x,x)
    strassenRecurse(x,x)

    # when btime ing use $A etc)
    #C2=@time strassenNoRecurse(A,B)
    C  = @time mult(A,B)

    C1 = @time naiveMult(A,B)
    C2 = @time strassenNoRecurse(A,B)
    C3 = @time blockedMult(A,B)
    #display(abs.(C3.-C).>=.00001)
    #C3=@time strassenRecurse(A,B)

    for i in 1:N
        for j in 1:N
            slowValid = max(slowValid, abs(C[i,j]-C1[i,j]))
            stupidValid = max(stupidValid, abs(C[i,j]-C2[i,j]))
            strassenValid = max(strassenValid, abs(C[i,j]-C3[i,j]))
        end
    end

    println(slowValid)
    println(stupidValid)
    println(strassenValid)
end
#testMults(5, Float64)
#println(getTimes(250:400,Int64,20,strassenNoRecurse))
#println(getTimes(250:400,Int64,20,mult))
#println(2 .^(5:10))
#println("Naive Mult")
#println(getTimes(1 .+ 2 .^(5:11),Int64,20,naiveMult))
#println("Recursive Base Mult")
#println(getTimes(2 .^(5:11),Int64,20,strassenRecurse))
#println("Recursive Naive Mult")
#println(getTimes(1 .+ 2 .^(5:11),Int64,20,strassenNaiveRecurse))
#println("One-Layer Base Mult")
#println(getTimes(2 .^(5:11),Int64,20,strassenNoRecurse))
#println("One-Layer Naive Mult")
#println(getTimes(1 .+ 2 .^(5:11),Int64,20,strassenNaiveNoRecurse))
#println("Base")
#println(getTimes(2 .^(5:11),Int64,20,mult))
println("Blocked Mult")
println(getTimes(1 .+ 2 .^(5:11),Int64,20,blockedMult))
println("One-Layer Blocked Mult")
println(getTimes(1 .+ 2 .^(5:11),Int64,20,strassenStripedNoRecurse))
println("Recursive Blocked Mult")
println(getTimes(1 .+ 2 .^(5:11),Int64,20,strassenStripedRecurse))
#println("Recursive 16")
#println(getTimes(2 .^(5:11),Int64,20,strassenRecurse16))
#println("Recursive 16 Naive Mult")
#println(getTimes(2 .^(5:11),Int64,20,strassenRecurse16Naive))
#println("Recursive 340 Base Mult")
#println(getTimes(2 .^(5:11),Int64,20,strassenOptimalRecurse))
#println("Floating points:")
#println("base")
#println(getTimes(2 .^(5:11),Float64,20,mult))
#println("One-layer")
#println(getTimes(2 .^(5:11),Float64,20,strassenNoRecurse))
#println("Recursive")
#println(getTimes(2 .^(5:11),Float64,20,strassenRecurse))
