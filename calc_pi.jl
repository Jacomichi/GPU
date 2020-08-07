using CuArrays
using BenchmarkTools

function dcalc_pi(n)
  4 * sum(CuArrays.rand(Float64,n) .^2 .+ CuArrays.rand(Float64,n) .^2 .<= 1) / n
end

function hcalc_pi(n)
  inside = 0
  for i in 1:n
    x, y = rand(),rand()
    inside += (x^2 + y^2 <= 1)
  end
  return 4 * inside/n
end

a = hcalc_pi(10_000_000)
b = dcalc_pi(10_000_000)
println(a)
println(b)
