using CUDAdrv, CUDAnative,CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
end

#2DGrid & 2Dblock
function dadd_mat2DG2DB(a,b,c,nx,ny)
  #@cuprintf("nx:%d ny:%d length \n",nx,ny,length(a))
  ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  #ここは配列が1から始まるので、1引いておく
  idx = (iy - 1) * nx + ix

  #Juliaの整数型は64bitsなので、printfでとるときは%ldになる。Cの整数型は32bitsなので%dになる。
  #@cuprintf("threadIdx:(%ld,%ld,%ld)blockIdx:(%ld,%ld,%ld)blockDim:(%ld,%ld,%ld)gridDim:(%ld,%ld,%ld)\n",
  #threadIdx().x,threadIdx().y,threadIdx().z,blockIdx().x,blockIdx().y,blockIdx().z,
  #blockDim().x,blockDim().y,blockDim().z,gridDim().x,gridDim().y,gridDim().z)
  #@cuprintf("ix:%ld iy:%ld nx:%ld idx:%ld\n",ix,iy,nx,idx)
  if ix <= nx && iy <= ny
    c[idx] = a[idx] + b[idx]
  end
  return nothing
end

function transposeNaiveRow(output,input,nx,ny)
  ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if ix <= nx && iy <= ny
    output[(ix - 1) * ny + iy] = input[(iy - 1) * nx + ix]
  end

  return nothing
end

function transposeNaiveCol(output,input,nx,ny)
  ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if ix <= nx && iy <= ny
    output[(iy - 1) * nx + ix] = input[(ix - 1) * ny + iy]
  end

  return nothing
end


function bench_transposeNaiveCol(output,input,nx,ny,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks transposeNaiveCol(output,input,nx,ny)
    end
end

function bench_transposeNaiveRow(output,input,nx,ny,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks transposeNaiveRow(output,input,nx,ny)
    end
end


nx = 2^11
ny = 2^11
mat_dims = (ny,nx)
a = rand(mat_dims...)
b = zeros(nx,ny)

d_a = CuArray(a)
d_b = CuArray(b)

numthreads = (2^5 ,2^5)
numblocks = ceil.(Int,mat_dims ./ numthreads)

println("threads:$(numthreads) , blocks:$(numblocks)")
setting = Config(numthreads,numblocks)
@btime bench_transposeNaiveCol(d_b,d_a,nx,ny,setting)

#d_b = transpose(d_a)
#println(d_b)
#@cuda threads=numthreads blocks=numblocks transposeNaiveRow(d_b,d_a,nx,ny)
#@cuda threads=numthreads blocks=numblocks transposeNaiveCol(d_b,d_a,nx,ny)

b = transpose(a)
#println(b)
@show (@test all(Array(d_b) .== b))
