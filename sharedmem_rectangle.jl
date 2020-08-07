using CUDAdrv, CUDAnative,CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
  shmem
end

function setRowReadCol(output)
  idx = (threadIdx().y - 1) * blockDim().x + threadIdx().x

  #転置した行列の用の配列の座標を作る・
  #配列が1から始まるので、少し変な感じになっている。
  #上手いやり方ないのかな、、、
  irow = (idx - 1) ÷ blockDim().y + 1
  icol = (idx - 1) % blockDim().y + 1
  #@cuprintf("col : %ld, row : %ld\n",icol,irow)
  tile = @cuStaticSharedMem(Int64,(16,32))

  tile[threadIdx().y,threadIdx().x] = idx

  CUDAnative.sync_threads()

  output[idx] = tile[icol,irow]

  return nothing
end

function bench_setRowReadCol(output,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks setRowReadCol(output)
    end
end

function setRowReadColDynamic(output)
  idx = (threadIdx().y - 1) * blockDim().x + threadIdx().x
  irow = (idx - 1) ÷ blockDim().y + 1
  icol = (idx - 1) % blockDim().y + 1

  tile = @cuDynamicSharedMem(Int64, (blockDim().y,blockDim().x))

  tile[threadIdx().y,threadIdx().x] = idx

  CUDAnative.sync_threads()

  output[idx] = tile[icol,irow]

  return nothing
end

function bench_setRowReadColDynamic(output,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks shmem=setting.shmem setRowReadColDynamic(output)
    end
end

nx = 32
ny = 16

d_out = CuArrays.zeros(Int64,ny,nx)#行列では、縦から始まるのでY,Xの順番
numthreads = (nx,ny)#threadの呼び出しは一つ目の引数がx、2つめの引数がyになっている。
numblocks = (1,1)
numshmem =  2 * prod(numthreads) * sizeof(Int64)
#@cuda threads=numthreads blocks=numblocks shmem=numshmem setRowReadColDynamic(d_out)


setting = Config(numthreads,numblocks,numshmem)

println("Static")
@btime bench_setRowReadCol(d_out,setting)

println("Dynamics")
@btime bench_setRowReadColDynamic(d_out,setting)

#=
Static
  15.531 μs (30 allocations: 832 bytes)
Dynamics
  15.509 μs (28 allocations: 752 bytes)
=#
