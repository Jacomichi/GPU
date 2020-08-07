using CUDAdrv, CUDAnative,CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
end

function hadd_mat(a,b,c)
  ny,nx = size(a)
  for i in 1:ny
    for j in 1:nx
      c[i,j] = a[i,j] + b[i,j]
    end
  end
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


function bench_dadd_mat2DG2DB(a,b,c,setting::Config,mat_dim)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks dadd_mat2DG2DB(a,b,c,mat_dim...)
    end
end

dimx,dimy = parse.(Int,ARGS[1:2])


N = 2^14
mat_dims = (N,N)
a = rand(mat_dims...)
b = rand(mat_dims...)
c = similar(a)


d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)

c = a .+ b

#2DG2DB
numthreads = (dimx,dimy)
numblocks = ceil.(Int,mat_dims./numthreads)


println("threads:$(numthreads) , blocks:$(numblocks)")
#@cuda threads=numthreads blocks=numblocks dadd_mat1DG1DB(d_a,d_b,d_c,mat_dims...)

setting = Config(numthreads,numblocks)
@btime bench_dadd_mat2DG2DB(d_a,d_b,d_c,setting,mat_dims)

@show (@test all(Array(d_c) .== c))
