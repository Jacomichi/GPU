using CUDAnative,CuArrays
using Random: rand!
using Parameters

using SimpleHeatmaps
using PyPlot

function shrinkshow(s,ratio::Int=4)
  r = ratio
  ss = Array(s)
  m,n = size(ss)
  @assert mod(m,r) == 0
  @assert mod(n,r) == 0

  shrinked = r^2 \  reshape(sum(sum(reshape(ss,(r,m÷r,r,n÷r)),dims=1),dims=3),(m÷r,n÷r))

  matshow(@. 2 \ (shrinked + 1))
end

function plot_ising2d(fig,s,ratio::Int=4)
  r = ratio
  ss = Array(s)
  m,n = size(ss)
  shrinked = r^2 \  reshape(sum(sum(reshape(ss,(r,m÷r,r,n÷r)),dims=1),dims=3),(m÷r,n÷r))
  m,n = size(shrinked)
  pcolormesh(0:m,0:n,shrinked,vmin=-2.0,vmax=1.0,cmap="gist_earth")
  fig[:set_aspect]("equal")
end

struct IsingPlan
  rnd
  m
  n
  β::Float64
  threads
  blocks
end

function genplan(m=1024,n=1024,β=log(1+sqrt(2)))
  rnd = CuArrays.rand(m,n)
  s = @. Int64(2*(rnd < 0.5) -1)
  β= Float64(β)
  threads = (32,32)
  blocks = @. ceil(Int,(m,n)/threads)

  (s,IsingPlan(rnd,m,n,β,threads,blocks))
end

function _update!(s,rnd,m,n,β,white=false)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  if i ≤m && j ≤n && iseven(i+j) == white
    @inbounds ajs =(s[ifelse(i+1≤m,i+1,1),j] +
                    s[ifelse(i-1≥1,i-1,m),j] +
                    s[i,ifelse(j+1≤n,j+1,1)] +
                    s[i,ifelse(j-1≥1,j-1,n)])
    @inbounds prob = CUDAnative.exp.(-β*s[i,j] * ajs)
    @inbounds s[i,j] = ifelse(rnd[i,j] < prob,-s[i,j],s[i,j])
  end

  return
end

function update!(s,iplan)
  @unpack rnd,m,n,β,threads,blocks = iplan
  rand!(rnd)
  @cuda blocks=blocks threads=threads _update!(s,rnd,m,n,β,true)
  @cuda blocks=blocks threads=threads _update!(s,rnd,m,n,β,false)
  nothing
end

s,iplan = genplan(2^12,2^12,2.0)
update!(s,iplan)

figure(figsize = (2^4,2^3))
fig = subplot(121);plot_ising2d(fig,s,8);title("t=0")
@time for i in 1:10000
  update!(s,iplan)
end
fig = subplot(122);plot_ising2d(fig,s,8);title("t = \$t_{fin}\$")
tight_layout()
savefig("Ising2d_orderT2.0.png")
#shrinkshow(s,8)

#=
anim = @time openanim(:gif) do aio
  for i in 1:10
    for j in 1:16
      update!(s,iplan)
    end
    img = shrinkshow(s,16)
    addframe(aio,img)
  end
end
=#

