using CUDAdrv, CUDAnative, CuArrays
using Random

function haversine_cpu(lat1::Float64,lon1::Float64,lat2::Float64,lon2::Float64,radius::Float64)
  c1 = cospi(lat1 / 180.0e0)
  c2 = cospi(lat2 / 180.0e0)
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  d1 = sinpi(dlat / 360.0e0)
  d2 = sinpi(dlon / 360.0e0)
  t = d2 * d2 * c1 * c2
  a =d1 * d1 + t
  c = 2.0e0 * asin(min(1.0e0,sqrt(a)))
  return radius * c
end

function pairwise_dis_cpu(lat::Vector{Float64},lon::Vector{Float64})
  n = length(lat)
  rowresult = Array{Float64}(undef,n,n)

  for i in 1:n,j in 1:n
    @inbounds rowresult[i,j] = haversine_cpu(lat[i],lon[i],lat[j],lon[j],6372.8e0)
  end

  return rowresult
end

function harversine_gpu(lat1::Float64,lon1::Float64,lat2::Float64,lon2::Float64,radius::Float64)
  c1 = CUDAnative.cospi(lat1 / 180.0e0)
  c2 = CUDAnative.cospi(lat2 / 180.0e0)
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  d1 = CUDAnative.sinpi(dlat / 360.0e0)
  d2 = CUDAnative.sinpi(dlon / 360.0e0)
  t = d2 * d2 * c1 * c2
  a = d1 * d1 + t
  c = 2.0e0 * CUDAnative.asin(CUDAnative.min(1.0e0,CUDAnative.sqrt(a)))
  return radius * c
end

function pairwise_dist_kernel(lat::CuDeviceVector{Float64},lon::CuDeviceVector{Float64},rowresult::CuDeviceMatrix{Float64},n)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if i <= n && j <= n
    shmem = @cuDynamicSharedMem(Float64, 2*blockDim().x + 2 * blockDim().y)
    if threadIdx().y == 1
      shmem[threadIdx().x] = lat[i]
      shmem[blockDim().x + threadIdx().x] = lon[i]
    end
    if threadIdx().x == 1
      shmem[2 * blockDim().x + threadIdx().y] = lat[j]
      shmem[2 * blockDim().x + blockDim().y + threadIdx().y] = lon[j]
    end
    sync_threads()

    lat_i = shmem[threadIdx().x]
    lon_i = shmem[blockDim().x + threadIdx().x]
    lat_j = shmem[2 * blockDim().x + threadIdx().y]
    lon_j = shmem[2 * blockDim().x + blockDim().y + threadIdx().y]

    @inbounds rowresult[i,j] = harversine_gpu(lat_i,lon_i,lat_j,lon_j,6372.8e0)
  end
  return
end

function pairwise_dist_gpu(lat::Vector{Float64},lon::Vector{Float64})
  lat_gpu = CuArray(lat)
  lon_gpu = CuArray(lon)

  n = length(lat)
  rowresult_gpu = CuArray(zeros(Float64,n,n))

  function get_config(kernel)


    function get_threads(threads)
      threads_x = floor(Int,sqrt(threads))
      threads_y = threads ÷threads_x
      return (threads_x,threads_y)
    end

    get_shmem(threads) = 2 * sum(threads) * sizeof(Float64)

    fun = kernel.fun
    config = launch_configuration(fun,shmem=threads->get_shmem(get_threads(threads)))

    threads = get_threads(config.threads)
    blocks = ceil.(Int, n./threads)
    shmem = get_shmem(threads)

    return (threads=threads,blocks=blocks,shmem=shmem)
  end

  @cuda config=get_config pairwise_dist_kernel(lat_gpu,lon_gpu,rowresult_gpu,n)

  return Array(rowresult_gpu)
end

using Test

function main(n = 1000)
  lat = rand(Float64,n) .* 45
  lon = rand(Float64,n) .* -120

  @test pairwise_dis_cpu(lat,lon) ≈pairwise_dist_gpu(lat,lon) rtol =1e-2
end

main()
