using CUDAdrv, CUDAnative,CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
  shmem
end

function transposeNaiveRow(output,input,nx,ny)
  ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if ix <= nx && iy <= ny
    output[(ix - 1) * ny + iy] = input[(iy - 1) * nx + ix]
  end

  return nothing
end

function bench_transposeNaiveRow(output,input,nx,ny,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks transposeNaiveRow(output,input,nx,ny)
    end
end


function transposeShmem(output,input,nx,ny)

  tile = @cuDynamicSharedMem(Float64, (blockDim().y,blockDim().x))
  ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  #元の行列の線形インデックス
  ti = (iy - 1) * nx + ix

  #転置ブロックのスレッドインデックス
  bidx = (threadIdx().y - 1) * blockDim().x + threadIdx().x
  #divremでも行ける?
  irow = (bidx - 1) ÷ blockDim().y + 1
  icol = (bidx - 1) % blockDim().y + 1

  #転置行列の座標(スレッドの座標)
  #blockIdxの部分がxとyで元のと入れ替わっている。
  #icolとirowを足している。
  ix = (blockIdx().y - 1) * blockDim().y + icol
  iy = (blockIdx().x - 1) * blockDim().x + irow

  #転置行列の線形インデックス(グローバルインデックス)
  to = (iy - 1) * ny + ix

  if ix <= nx && iy <= ny
    tile[threadIdx().y,threadIdx().x] = input[ti]
    CUDAnative.sync_threads()
    output[to] = tile[icol,irow]
  end

  return nothing
end

function bench_transposeShmem(output,input,nx,ny,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks shmem=setting.shmem transposeShmem(output,input,nx,ny)
    end
end


nx = 2^12
ny = 2^12

mat_dims = (ny,nx)
a = rand(mat_dims...)
b = zeros(nx,ny)

d_a = CuArray(a)
d_b = CuArray(b)

numthreads = (2^5 ,2^4)
numblocks = ceil.(Int,mat_dims ./ numthreads)
numshmem =  2 * prod(numthreads) * sizeof(Float64)

println("threads:$(numthreads) , blocks:$(numblocks)")
setting = Config(numthreads,numblocks,numshmem)
println("naive")
@btime bench_transposeNaiveRow(d_b,d_a,nx,ny,setting)

println("shared memory")
@btime bench_transposeShmem(d_b,d_a,nx,ny,setting)
#@cuda threads=numthreads blocks=numblocks shmem=numshmem transposeShmem(d_b,d_a,nx,ny)

b = transpose(a)

@show (@test all(Array(d_b) .== b))

#=
threads:(16, 16) , blocks:(128, 128)
naive
  890.935 μs (61 allocations: 1.84 KiB)
shared memory
  1.324 ms (63 allocations: 1.92 KiB)


threads:(16, 16) , blocks:(256, 256)
  naive
    3.455 ms (61 allocations: 1.84 KiB)
  shared memory
    4.986 ms (63 allocations: 1.92 KiB)

threads:(32, 16) , blocks:(128, 256)
naive
  3.528 ms (61 allocations: 1.84 KiB)
shared memory
  4.910 ms (63 allocations: 1.92 KiB)

threads:(16, 16) , blocks:(512, 512)
  naive
    16.330 ms (61 allocations: 1.84 KiB)
  shared memory
    20.923 ms (63 allocations: 1.92 KiB)
=#
