using CUDAdrv, CUDAnative,CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
end


function dvadd(a,b,c)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  c[i] = a[i] + b[i]
  return
end

function bench_dvadd(a,b,c,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks dvadd(a,b,c)
    end
end

function hvadd(a,b,c)
  for i in eachindex(a)
    c[i] = a[i] + b[i]
  end
  return
end

N = 2^24
a = rand(N)
b = rand(N)
c = similar(a)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)


numthreads = 2^10
numblocks = ceil(Int,N/numthreads)
setting = Config(numthreads,numblocks)
#@cuda threads=numthreads blocks = numblocks dvadd(d_a,d_b,d_c)
#hvadd(a,b,c)
#@show (@test all(Array(d_c) .== c))
CUDAdrv.@profile bench_dvadd(d_a,d_b,d_c,setting)
