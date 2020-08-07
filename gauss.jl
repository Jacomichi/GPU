using CuArrays, CUDAnative, CUDAdrv
using Test, BenchmarkTools

function gpu_gauss(x,y)
    CUDAnative.exp.(x .* x + y .* y)
end

function gpu_cal_gauss(val)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(x)*length(x)
        x = (i / n - n/2) * (l/n)
        y = (i % n - n/2) * (l/n)
        val[i] = CUDAnative.exp(-(x * x + y * y))
    end
end

N = 2^10
x_d = CuArray(collect(0:N))
L = 10
sum_x = sum(CUDAnative.exp.(-L.^2 .* (x_d./N .- (0.5)).^2))/N *L
println(sum_x^2)
