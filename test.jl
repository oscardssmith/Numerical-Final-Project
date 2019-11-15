using LinearAlgebra
using BenchmarkTools
using Random

# This shouldn't fix things, but does
promote_op = Base.promote_op

matprod(x, y) = x*y + x*y

function mult(A::AbstractMatrix, B::AbstractMatrix)
    @boundscheck size(A,2) == size(B,1)
    A*B
end

function naiveMult(A::AbstractMatrix,B::AbstractMatrix)
    @boundscheck size(A,2) == size(B,1)
    TS = promote_op(matprod, eltype(A), eltype(B))
    C = zeros(TS, size(A,1), size(B,2))
    return naiveMult!(C,A,B)
end

function naiveMult!(C,A,B)
    """ Fastest for ~nxn*nx16 for n in [50,75] """
    @inbounds for i in 1:size(A,1)
        for j in 1:size(B,2)
            @simd for z in 1:size(B,1)
                C[i,j] += A[i,z]*B[z,j]
            end
        end
    end
    return C
end

function blockedMult(A::AbstractMatrix,B::AbstractMatrix)
    @boundscheck size(A,2) == size(B,1)
    TS = promote_op(matprod, eltype(A), eltype(B))
    C = zeros(TS, size(A,1), size(B,2))
    return blockedMult!(A,B,C)
end

function blockedMult!(A, B, C)
    bs = 16
    n = size(A,1)
    if n<bs
        return naiveMult!(C,A,B)
    end
    @inbounds for kk in 1:bs:n               # iterates over cols of A (rows of B)
        for jj in 1:bs:n                     # iterates over rows of B

            for i in 1:n                     # pick slice A[i,kk:kk+bs]
               for j in jj:jj+bs-1           # Make dot product  A[i,kk:kk+bs] * B_block[:,j] for all j
                   for k in kk:kk+bs-1
                       C[i,j] += A[i,k] * B[k,j]
                   end
               end
            end
        end
    end
    return C
end

function pad!(A,B,m,n,p)
    if p%2 == 1
        if n%2 ==1
            Anew=zeros(eltype(A), n+1,p+1)
            Anew[1:end-1,1:end-1] .= A
            A = Anew
        else
            Anew=zeros(eltype(A), n,p+1)
            Anew[1:end,1:end-1] .= A
            A = Anew
        end
        if m%2==1
            Bnew=zeros(eltype(B), p+1, m+1)
            Bnew[1:end-1,1:end-1] .= B
            B = Bnew
        else
            Bnew=zeros(eltype(B), p+1, m)
            Anew[1:end,1:end-1] .= B
            B = Bnew
        end
    else
        if n%2 ==1
            Anew=zeros(eltype(A), n+1, p)
            Anew[1:end-1,1:end] .= A
            A = Anew
        end
        if m%2==1
            Bnew=zeros(eltype(B), p, m+1)
            Bnew[1:end,1:end-1] .= B
            B = Bnew
        end
    end
    return A,B
end

function strassenBase(A::AbstractMatrix,B::AbstractMatrix, mult)
    @boundscheck size(A,2) == size(B,1)
    n, p, m = size(A,1), size(B,1), size(B,2)

    a1,a2 = (div(n,2),div(p,2))
    b1,b2 = (div(p,2),div(m,2))

    A, B = pad!(A,B,m,n,p)
    TS = promote_op(matprod, eltype(A), eltype(B))
    C = zeros(TS, size(A,1), size(B,2))
    return strassenBase!(C,A,B,mult)[1:n, 1:m]
end

function strassenBase!(C:: AbstractMatrix, A::AbstractMatrix,B::AbstractMatrix, mult)
    """ Takes in pre-padded arrays"""

    n, p, m = size(A,1), size(B,1), size(B,2)

    a1,a2 = (div(n,2),div(p,2))
    b1,b2 = (div(p,2),div(m,2))
    @inbounds @views begin
        A00 = A[1:a1,1:a2]
        A10 = A[a1+1:end,1:a2]
        A01 = A[1:a1,a2+1:end]
        A11 = A[a1+1:end,a2+1:end]

        B00 =B[1:b1,1:b2]
        B10 =B[b1+1:end,1:b2]
        B01 =B[1:b1,b2+1:end]
        B11=B[b1+1:end,b2+1:end]

        M0=mult(A00.+A11, B00.+B11)
        M1=mult(A10.+A11, B00)
        M2=mult(A00,      B01.-B11)
        M3=mult(A11,      B10.-B00)
        M4=mult(A00.+A01, B11)
        M5=mult(A10.-A00, B00.+B01)
        M6=mult(A01.-A11, B10.+B11)

        C[1:a1,1:a2] .= M0.+M3.+M6.-M4
        C[a1+1:end,1:b2] .= M1.+M3
        C[1:a1,b2+1:end] .= M2.+M4
        C[a1+1:end,b2+1:end] .= M0.+M2.+M5.-M1
        return C
    end
end

function strassenNoRecurse(A::AbstractMatrix,B::AbstractMatrix)
    return strassenBase(A, B, mult)
end

function strassenRecurse(A::AbstractMatrix,B::AbstractMatrix)
    minSize = 501
    if(size(A,1)<minSize||size(A,2)<minSize||size(B,1)<minSize||size(B,2)<minSize)
        return mult(A,B)
    else
        return strassenBase(A,B,strassenRecurse)
    end
end


function testMults(N::Int64,T)
    slowValid=1
    stupidValid=1
    strassenValid=1
    A = rand(T,N,N)
    B = rand(T,N,N)

    #precompile
    x = zeros(T,2,2)
    mult(x,x)
    naiveMult(x,x)
    blockedMult(x,x)

    C =@btime mult($A,$B)
    C =@btime blockedMult($A,$B)
    C1=@btime naiveMult($A,$B)
end

testMults(400, Int)
