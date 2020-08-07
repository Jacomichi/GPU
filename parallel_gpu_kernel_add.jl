using CuArrays, CUDAnative, CUDAdrv
using Test, BenchmarkTools

function gpu_add2!(y,x)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function bench_gpu2!(y,x)
    CuArrays.@sync begin
        @cuda threads=256 gpu_add2!(y,x)
    end
end

function gpu_add3!(y,x)
    #JuliaではthreadId()やblockIdxは1から始まるようになっている。
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

function bench_gpu3!(y,x)
    numblocks = ceil(Int,N/256)
    CuArrays.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y_d,x_d)
    end
end

function add_broadcast!(y,x)
    #CuArrays.@syncを用いるとCPUからGPUにジョブを投げることができる。
    CuArrays.@sync y.+=x
    return
end

function gpu_add2_println!(y,x)
    index = threadIdx().x
    stride = blockDim().x
    @cuprintln("thread $index, block $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end


N = 2^20
x_d = CuArrays.fill(1.0f0,N)
y_d = CuArrays.fill(2.0f0,N)

#@cuda threads=256 gpu_add2!(y_d,x_d)

#numblocks = ceil(Int,N/256)
#@cuda threads=256 blocks=numblocks gpu_add3!(y_d,x_d)
#@show (@test all(Array(y_d) .== 3.0f0))

#@btime bench_gpu3!(y_d,x_d)
#@btime add_broadcast!(y_d,x_d)

@cuda threads=16 gpu_add2_println!(y_d,x_d)


#Note that the printed output is only generated when synchronizing the entire GPU with synchronize().
#This is similar to CuArrays.@sync, and is the counterpart of cudaDeviceSynchronize in CUDA C++.
synchronize()
