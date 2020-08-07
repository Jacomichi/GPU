using CUDAdrv, CUDAnative,CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
end

function mathkernel(c)
  index = (blockIdx().x -1) * blockDim().x + threadIdx().x
  ia = ib = 0.0

  if index % 2 == 0
    ia = 100.0
  else
    ib = 200.0
  end
  c[index] = ia + ib
  return nothing
end

function mathkernel_warpsize(c)
  index = (blockIdx().x -1) * blockDim().x + threadIdx().x
  ia = ib = 0.0

  if (index/CUDAnative.warpsize()) % 2 == 0
    ia = 100.0
  else
    ib = 200.0
  end
  c[index] = ia + ib
  return nothing
end

function bench_mathkernel(c,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks mathkernel(c)
    end
end

function bench_mathkernel_warpsize(c,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks mathkernel_warpsize(c)
    end
end

N = 64
numthreads = 64
numblocks = 1
setting = Config(numthreads,numblocks)
d_c = CuArrays.zeros(N)

println("Normal")
@btime bench_mathkernel(d_c,setting)

d_c = CuArrays.zeros(N)
println("Divergence")
@btime bench_mathkernel_warpsize(d_c,setting)

#println(d_c)
