using CuArrays, CUDAnative, CUDAdrv
using Test, BenchmarkTools

function check_order_CuArray(a)
    idx = threadIdx().x
    @cuprintf("a[%ld] = %ld from threadIdx %ld\n",idx,a[idx],idx)
    return nothing
end

N = 2^5
a = reshape(collect(1:N),2,ceil(Int,N/2))
d_a = CuArray(a)
println(a)
@cuda threads=N check_order_CuArray(d_a)
