using CuArrays, CUDAnative, CUDAdrv

function cuadd!(y,x)
  index = threadIdx().x
  stride = blockDim().x
  for i = index:stride:length(y)
    @inbounds y[i] += x[i]
  end
  return nothing
end

N = 100_000
a = CuArrays.fill(1.0f0,N)
b = CuArrays.fill(2.0f0,N)
@cuda threads=256 cuadd!(b,a)
println(sum(b))
c = a.+b
println(sum(c))
