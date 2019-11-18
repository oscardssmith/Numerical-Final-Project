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
    println("b")
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
testMults(800, Int)
#getTimes(200:210,Int64,strassenRecurse)

#println(getTimes(2 .^(3:10),Int64,20,strassenRecurse))

