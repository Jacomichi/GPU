using CUDAdrv, CUDAnative, CuArrays
using Test, BenchmarkTools

function gpu_check_global_variable(a)
    idx = threadIdx().x
    @cuprintf("%ld\n",a[idx])
    a[idx] += 1
    @cuprintf("%ld\n",a[idx])
    return nothing
end

a = 2
d_a = CuArray([a])
println(d_a)
@cuda threads=1 gpu_check_global_variable(d_a)
println(d_a)
