using CUDAdrv, CUDAnative, CuArrays
using Test, BenchmarkTools

function nested_Hello_World(iSize,iDepth)

  tid = threadIdx().x
  @cuprintf("Recursion=%ld: Hello World from thread %ld block %ld\n", iDepth, tid,blockIdx().x)

  if iSize == 1
    return nothing
  end

  nthreads = iSize >> 1

  if tid == 1 && nthreads > 0
    @cuda dynamic=true threads=nthreads nested_Hello_World(nthreads,iDepth + 1)
    @cuprintf("-------> nested execution depth: %ld\n", iDepth + 1)
  end
  return nothing
end

function nested_Hello_World2(iSize,iDepth)

  tid = threadIdx().x
  @cuprintf("Recursion=%ld: Hello World from thread %ld block %ld\n", iDepth, tid,blockIdx().x)

  if iSize == 1
    return nothing
  end

  nthreads = iSize >> 1

  #この部分が上とは違う
  if tid == 1 && nthreads > 0 && blockIdx().x == 1
    @cuda dynamic=true threads=nthreads blocks=2 nested_Hello_World2(nthreads,iDepth + 1)
    @cuprintf("-------> nested execution depth: %ld\n", iDepth + 1)
  end
  return nothing
end


numthreads = 2^3
@cuda threads=numthreads blocks=2 nested_Hello_World2(numthreads,0)
