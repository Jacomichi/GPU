using CUDAdrv, CUDAnative, CuArrays

function apply(op,cu_arr)
  i = threadIdx().x
  cu_arr[i] = op(cu_arr[i])
  return
end

a = CuArray([1.,2.,3.])

@cuda threads=length(a) apply(x->x^2,a)
println(a)
println([1] .+ [2 2] .+ [3 3;3 3])
println(typeof([2 2]))
