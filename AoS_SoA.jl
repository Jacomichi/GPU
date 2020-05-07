using CuArrays, CUDAnative, CUDAdrv
using Test, BenchmarkTools

struct innerstruct
    x::Float64
    y::Float64
end

function innerstruct()
    innerstruct(rand(),rand())
end

struct innerArray
    x::CuArray{Float64,1}
    y::CuArray{Float64,1}
end

N = 2^4
myAOS = CuArray([innerstruct() for i in 1:N])
mySOA = innerArray(CuArrays.rand(N),CuArrays.rand(N))
twoDmat = CuArrays.rand(N,2)
println(myAOS)
println("-----")
println(mySOA)
println("-----")
println(twoDmat)
