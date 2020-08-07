using CUDAdrv, CUDAnative,CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
  shmem
end

function setRowReadRow(output)
  idx = (threadIdx().y - 1) * blockDim().x + threadIdx().x

  tile = @cuStaticSharedMem(Int64,(32,32))

  tile[threadIdx().y,threadIdx().x] = idx

  CUDAnative.sync_threads()

  output[idx] = tile[threadIdx().y,threadIdx().x]

  return nothing
end

function bench_setRowReadRow(output,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks setRowReadRow(output)
    end
end

function setColReadCol(output)
  idx = (threadIdx().y - 1) * blockDim().x + threadIdx().x

  tile = @cuStaticSharedMem(Int64,(32,32))

  tile[threadIdx().x,threadIdx().y] = idx

  CUDAnative.sync_threads()

  output[idx] = tile[threadIdx().x,threadIdx().y]

  return nothing
end

function bench_setColReadCol(output,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks setColReadCol(output)
    end
end

function setRowReadCol(output)
  idx = (threadIdx().y - 1) * blockDim().x + threadIdx().x

  tile = @cuStaticSharedMem(Int64,(32,32))

  tile[threadIdx().y,threadIdx().x] = idx

  CUDAnative.sync_threads()

  output[idx] = tile[threadIdx().x,threadIdx().y]

  return nothing
end

function bench_setRowReadCol(output,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks setRowReadCol(output)
    end
end

function setColReadColDynamic(output)
  idx = (threadIdx().y - 1) * blockDim().x + threadIdx().x

  tile = @cuDynamicSharedMem(Int64, (blockDim().y,blockDim().x))

  tile[threadIdx().x,threadIdx().y] = idx

  CUDAnative.sync_threads()

  output[idx] = tile[threadIdx().x,threadIdx().y]

  return nothing
end

function bench_setColReadColDynamic(output,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks shmem=setting.shmem setColReadColDynamic(output)
    end
end

function setRowReadColPading(output)
  idx = (threadIdx().y - 1) * blockDim().x + threadIdx().x

  #padingで1追加する。
  tile = @cuStaticSharedMem(Int64,(32,32 + 1))

  tile[threadIdx().y,threadIdx().x] = idx

  CUDAnative.sync_threads()

  output[idx] = tile[threadIdx().x,threadIdx().y]

  return nothing
end

function bench_setRowReadColPading(output,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks setRowReadColPading(output)
    end
end

d_out = CuArrays.zeros(Int64,32,32)
numthreads = (32,32)
numblocks = (1,1)
numshmem =  2 * prod(numthreads) * sizeof(Int64)
@cuda threads=numthreads blocks=numblocks shmem=numshmem setColReadColDynamic(d_out)


setting = Config(numthreads,numblocks,numshmem)
#=
println("RowReadRow")
@btime bench_setRowReadRow(d_out,setting)
println("ColReadCol")
@btime bench_setColReadCol(d_out,setting)

println("RowReadCol")
@btime bench_setRowReadCol(d_out,setting)
=#
println("RowReadColPading")
@btime bench_setRowReadColPading(d_out,setting)
#@btime bench_setColReadColDynamic(d_out,setting)
