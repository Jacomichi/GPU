using CUDAdrv, CUDAnative, CuArrays
using PyPlot

function julia_set!(z,c,count,iterate=100)
  i = (blockIdx().x -1 ) * blockDim().x + threadIdx().x
  for j in 1:iterate
    if abs2(z[i]) > 4
      count[i] = j
      return
    else
      z[i] = z[i]^2 + c[i]
    end
  end
  count[i] = iterate
  return
end

f(x,c) = x^2 + c


#Julia集合を計算する部分をCuArrayにしたい。まだ未完成
function arrjulia_set!(z,c,count,iterate=100)
  for j in 1:iterate
    if abs2.(z) .> 4
      count .= j
      return
    else
      z .=  z.^2 .+ c
    end
  end
end


function cujulia_set!(z_d,c_d,count_d,itr)
  @cuda threads=lensize[1] blocks=lensize[2] julia_set!(z_d,c_d,count_d,itr)
end

function plot_2d(fig,s)
    m,n = size(s)
    pcolormesh(0:m,0:n,s,cmap="gist_earth")
    fig[:set_aspect]("equal")
end

lensize = (2^9,2^9);pos = 0.01
c = zeros(Complex{Float64},lensize) .+ (-0.8 + 0.156im)
z = [Complex{Float64}(i +j*1im) for i in range(-pos,pos,length=lensize[1]), j in range(-pos,pos,length=lensize[2])]
z_d = CuArray(Array(transpose(z)))
c_d = CuArray(c)
count_d = CuArrays.zeros(Int64,lensize)

@time cujulia_set!(z_d,c_d,count_d,2^20)
figure(figsize = (2^4,2^4))
println("fin calc")
fig = subplot(121)
plot_2d(fig,count_d)
tight_layout()
savefig("Julia_set.png")
