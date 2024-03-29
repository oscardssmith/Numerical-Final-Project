using LinearAlgebra
using BenchmarkTools
using Random
using InteractiveUtils
import Base.Threads.@spawn

# This shouldn't fix things, but does
promote_op = Base.promote_op

matprod(x, y) = x*y + x*y
const bs=16
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
            s = C[i,j]
            @fastmath @simd for z in 1:size(B,1)
                s += A[i,z]*B[z,j]
            end
            C[i,j] = s
        end
    end
    return C
end

function blockedMult(A::AbstractMatrix,B::AbstractMatrix)

    @boundscheck size(A,2) == size(B,1)
    TS = promote_op(matprod, eltype(A), eltype(B))
    C = zeros(TS, size(A,1), size(B,2))
    Btemp = zeros(eltype(B),bs,bs)
    return blockedMult!(C, A, B, Btemp)
end


function blockedMult!(C, A, B, Btemp)
    n = size(A,1)
    left = n%bs
    @inbounds @fastmath for kk in 1:bs:n-left        # iterates over cols of A (rows of B)
        for jj in 1:bs:n-left                     # iterates over rows of B
            Btemp .= (@view B[kk:kk+bs-1,jj:jj+bs-1])'
            for i in 1:n-left                     # pick slice A[i,kk:kk+bs]
               for j in 1:bs           # Make dot product  A[i,kk:kk+bs] * B_block[:,j] for all j
                   s = C[i,j+jj-1]
                   @simd for k in 1:bs
                      s += A[i,k+kk-1] * Btemp[j,k]
                   end
                   C[i,j+jj-1] =s
               end
            end
        end
    end
    if left>0 #not a multiple of 16
        @inbounds @fastmath for j in 1:n-left
            for i in 1:n-left
                s = C[i,j]
                for k in n-left+1:n
                    s += A[i,k]*B[k,j]
                end
                C[i,j] = s
            end
        end
        @inbounds @fastmath for j in n-left+1:n
            for i in 1:n
                s = C[i,j]
                for k in 1:n
                    s += A[i,k]*B[k,j]
                end
                C[i,j] = s
            end
        end
        @inbounds @fastmath for j in 1:n-left
            for i in n-left+1:n
                s = C[i,j]
                for k in 1:n
                    s += A[i,k]*B[k,j]
                end
                C[i,j] = s
            end
        end
    end
    return C
end

function padAndSplit!(A,width,height)
    a1,a2=div(width+1,2),div(height+1,2)

    pad1=width%2
    pad2=height%2
    @views begin
        A00 = A[1:a1,1:a2]
        if pad1==0
            A10 = A[a1+1:end,1:a2]
        else
            A10=zeros(eltype(A),div(width+1,2),div(height+1,2))
            A10[1:end-pad1,1:end] .= A[a1+1:end,1:a2]
        end
        if pad2==0
            A01 = A[1:a1,a2+1:end]
        else
            A01=zeros(eltype(A),div(width+1,2),div(height+1,2))
            A01[1:end,1:end-pad2] .= A[1:a1,a2+1:end]
        end
        if pad1+pad2==0
            A11 = A[a1+1:end,a2+1:end]
        else
            A11=zeros(eltype(A),div(width+1,2),div(height+1,2))
            A11[1:end-pad1,1:end-pad2] .= A[a1+1:end,a2+1:end]
        end
        return A00,A10,A01,A11,zeros(eltype(A),div(width+1,2),div(height+1,2))
    end

end

function strassenBase(A::AbstractMatrix,B::AbstractMatrix, mult)
    @boundscheck size(A,2) == size(B,1)
    n, p, m = size(A,1), size(B,1), size(B,2)

    TS = promote_op(matprod, eltype(A), eltype(B))

    C = zeros(TS, n+n%2, m+m%2)
    return strassenBase!(C,A,B,mult)[1:n, 1:m]
end

function strassenBase!(C:: AbstractMatrix, A::AbstractMatrix,B::AbstractMatrix, mult)
    """ Takes in pre-padded arrays"""

    n, p, m = size(A,1), size(B,1), size(B,2)
    a1,a2 = (div(n+1,2),div(p+1,2))
    b1,b2 = (div(p+1,2),div(m+1,2))

    @inbounds @fastmath @views begin
        A00,A10,A01,A11,Atemp=padAndSplit!(A,n,p)
        B00,B10,B01,B11,Btemp=padAndSplit!(B,p,m)
        Atemp .= A00.+A11
        Btemp .= B00.+B11
        m0= @spawn mult(Atemp, Btemp)
        Atemp .= A10.+A11
        m1= @spawn mult(Atemp, B00)
        Btemp .= B01.-B11
        m2= @spawn mult(A00, Btemp)
        Btemp .=B10.-B00
        m3= @spawn mult(A11,Btemp)
        Atemp .= A00.+A01
        m4= @spawn mult(Atemp, B11)
        Atemp .= A10.-A00
        Btemp .= B00.+B01
        m5= @spawn mult(Atemp, Btemp)
        Atemp .= A01.-A11
        Btemp .= B10.+B11
        M6=  mult(Atemp, Btemp)
	M0 = fetch(m0)
	M1 = fetch(m1)
        M2 = fetch(m2)
        M3 = fetch(m3)
        M4 = fetch(m4)
	M5 = fetch(m5)

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
    minSize = 500
    if(size(A,1)<minSize||size(A,2)<minSize||size(B,1)<minSize||size(B,2)<minSize)
        return mult(A,B)
    else
        return strassenBase(A,B,strassenRecurse)
    end
end

function strassenStripedNoRecurse(A::AbstractMatrix,B::AbstractMatrix)
    return strassenBase(A, B, blockedMult)
end

function strassenStripedRecurse(A::AbstractMatrix,B::AbstractMatrix)
    minSize = 500
    if(size(A,1)<minSize||size(A,2)<minSize||size(B,1)<minSize||size(B,2)<minSize)
        return blockedMult(A,B)
    else
        return strassenBase(A,B,strassenStripedRecurse)
    end
end

function strassenNaiveNoRecurse(A::AbstractMatrix,B::AbstractMatrix)
    return strassenBase(A, B, naiveMult)
end

function strassenNaiveRecurse(A::AbstractMatrix,B::AbstractMatrix)
    minSize = 500
    if(size(A,1)<minSize||size(A,2)<minSize||size(B,1)<minSize||size(B,2)<minSize)
        return naiveMult(A,B)
    else
        return strassenBase(A,B,strassenNaiveRecurse)
    end
end

function strassenOptimalRecurse(A::AbstractMatrix,B::AbstractMatrix)
    minSize = 200
    if(size(A,1)<minSize||size(A,2)<minSize||size(B,1)<minSize||size(B,2)<minSize)
        return mult(A,B)
    else
        return strassenBase(A,B,strassenOptimalRecurse)
    end
end

function strassenRecurse16(A::AbstractMatrix,B::AbstractMatrix)
    minSize = 16
    if(size(A,1)<minSize||size(A,2)<minSize||size(B,1)<minSize||size(B,2)<minSize)
        return mult(A,B)
    else
        return strassenBase(A,B,strassenRecurse16)
    end
end

function strassenRecurse16Naive(A::AbstractMatrix,B::AbstractMatrix)
    minSize = 16
    if(size(A,1)<minSize||size(A,2)<minSize||size(B,1)<minSize||size(B,2)<minSize)
        return naiveMult(A,B)
    else
        return strassenBase(A,B,strassenRecurse16Naive)
    end
end
