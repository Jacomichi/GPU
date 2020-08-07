using CUDAdrv, CUDAnative, CuArrays
using Test, BenchmarkTools

struct Config
  threads
  blocks
end


function reduceNeighbored(input,output,max_itr)
  tid = threadIdx().x
  blockIdx_x_Dim = (blockIdx().x - 1 ) * blockDim().x
  idx = blockIdx_x_Dim + threadIdx().x

  if idx >= CuArrays.length(input)
    return
  end

  stride = 1
  for i in 1:max_itr
    #@cuprintf("stride %ld\n",stride)

    #tidは1から始まるので、ここは余りが1になる時を取り出す。
    if (tid % (2 * stride)) == 1
      #教科書では、inputのglobal pointer(全ての配列全体での位置)をlocal pointer(blockの内での位置)
      #にキャストしているので、indexにはtid(block内のでの位置)を用いているが、
      #Juliaでのlocal pointerへのキャストが分からなかったので、global pointerを使っている。
      input[idx] += input[idx + stride]
    end

    CUDAnative.sync_threads()

    stride *= 2
  end

  #blockにおける一番始めの要素に、そのblocknの要素の和が入っている。
  if tid == 1
    #idx = 1 + blockIdx_x_Dimの時にしか、この中に入らないので、idxで良い。
    output[blockIdx().x] = input[idx]
    #@cuprintf("output[%ld] = %ld\n",blockIdx().x,output[blockIdx().x])
  end

  return nothing
end


function reduceNeighbored2(input,output,max_itr)
  tid = threadIdx().x
  blockIdx_x_Dim = (blockIdx().x - 1 ) * blockDim().x
  idx = blockIdx_x_Dim + threadIdx().x
  #@cuprintf("blockDim : %ld\n",blockDim().x)
  if idx >= CuArrays.length(input)
    return
  end

  stride = 1
  for i in 1:max_itr
    #この部分がreduceNeighboredに比べて新しい。
    #スレッドIDを配列のローカルindexに変換
    #この変換ではindexが1から始まるように教科書とは、形を変えている
    index = 2 * stride * (tid - 1) + 1
    if index <= blockDim().x
      #@cuprintf("%ld\n",index)
      input[index + blockIdx_x_Dim] += input[index + blockIdx_x_Dim + stride]
    end

    CUDAnative.sync_threads()

    stride *= 2
  end

  if tid == 1
    output[blockIdx().x] = input[idx]
    #@cuprintf("output[%ld] = %ld\n",blockIdx().x,output[blockIdx().x])
  end

  return nothing
end


function reduceInterleaved(input,output,max_itr)
  tid = threadIdx().x
  blockIdx_x_Dim = (blockIdx().x - 1 ) * blockDim().x
  idx = blockIdx_x_Dim + threadIdx().x
  #@cuprintf("blockDim : %ld\n",blockDim().x)
  if idx >= CuArrays.length(input)
    return
  end

  #インターリーブ:strideをどんどん半分にしていく。
  #一番最初はblockのサイズの半分から
  stride = ceil(Int,blockDim().x / 2)
  for i in 1:max_itr
    #一番小さいstrideは1なので、条件に=も含める。
    if tid <= stride
      #@cuprintf("%ld\n",index)
      input[tid + blockIdx_x_Dim] += input[tid + blockIdx_x_Dim + stride]
    end

    CUDAnative.sync_threads()

    #shift演算:一桁ビットをずらすごとに値が半分になる。
    stride >>= 1
  end

  if tid == 1
    output[blockIdx().x] = input[idx]
    #@cuprintf("output[%ld] = %ld\n",blockIdx().x,output[blockIdx().x])
  end

  return nothing
end

function reduceInterleaved_unroll(input,output,max_itr)
  tid = threadIdx().x
  #ここを倍にしておく
  blockIdx_x_Dim = (blockIdx().x - 1 ) * blockDim().x * 2
  idx = blockIdx_x_Dim + threadIdx().x
  #@cuprintf("blockDim : %ld\n",blockDim().x)

  #2つのデータブロックを展開
  #リダクションする前に隣り合うブロックを足し上げる。
  if idx + blockDim().x <= CuArrays.length(input)
    input[idx] += input[idx + blockDim().x]
  end

  CUDAnative.sync_threads()

  stride = ceil(Int,blockDim().x / 2)
  for i in 1:max_itr
    #一番小さいstrideは1なので、条件に=も含める。
    if tid <= stride
      #@cuprintf("%ld\n",index)
      input[tid + blockIdx_x_Dim] += input[tid + blockIdx_x_Dim + stride]
    end

    CUDAnative.sync_threads()

    #shift演算:一桁ビットをずらすごとに値が半分になる。
    stride >>= 1
  end

  if tid == 1
    output[blockIdx().x] = input[idx]
    #@cuprintf("output[%ld] = %ld\n",blockIdx().x,output[blockIdx().x])
  end

  return nothing
end



function reduceInterleaved_unroll4(input,output,max_itr)
  tid = threadIdx().x
  #ここを倍にしておく
  blockIdx_x_Dim = (blockIdx().x - 1 ) * blockDim().x * 4
  idx = blockIdx_x_Dim + threadIdx().x
  #@cuprintf("blockDim : %ld\n",blockDim().x)

  #2つのデータブロックを展開
  #リダクションする前に隣り合うブロックを足し上げる。
  if idx + 3 * blockDim().x <= CuArrays.length(input)
    #=
    a1 = input[idx]
    a2 = input[idx + blockDim().x]
    a3 = input[idx + 2 * blockDim().x]
    a4 = input[idx + 3 * blockDim().x]
    input[idx] = a1 + a2 + a3 + a4
    =#
    input[idx] = input[idx] + input[idx + blockDim().x] + input[idx + 2 * blockDim().x] +input[idx + 3 * blockDim().x]
  end

  CUDAnative.sync_threads()

  stride = ceil(Int,blockDim().x / 2)
  for i in 1:max_itr
    #一番小さいstrideは1なので、条件に=も含める。
    if tid <= stride
      #@cuprintf("%ld\n",index)
      input[tid + blockIdx_x_Dim] += input[tid + blockIdx_x_Dim + stride]
    end

    CUDAnative.sync_threads()

    #shift演算:一桁ビットをずらすごとに値が半分になる。
    stride >>= 1
  end

  if tid == 1
    output[blockIdx().x] = input[idx]
    #@cuprintf("output[%ld] = %ld\n",blockIdx().x,output[blockIdx().x])
  end

  return nothing
end

function reduceInterleaved_unroll8(input,output,max_itr)
  tid = threadIdx().x
  #ここを倍にしておく
  blockIdx_x_Dim = (blockIdx().x - 1 ) * blockDim().x * 8
  idx = blockIdx_x_Dim + threadIdx().x
  #@cuprintf("blockDim : %ld\n",blockDim().x)

  #2つのデータブロックを展開
  #リダクションする前に隣り合うブロックを足し上げる。
  if idx + 7 * blockDim().x <= CuArrays.length(input)
    a1 = input[idx]
    a2 = input[idx + blockDim().x]
    a3 = input[idx + 2 * blockDim().x]
    a4 = input[idx + 3 * blockDim().x]
    a5 = input[idx + 4 * blockDim().x]
    a6 = input[idx + 5 * blockDim().x]
    a7 = input[idx + 6 * blockDim().x]
    a8 = input[idx + 7 * blockDim().x]
    input[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
  end

  CUDAnative.sync_threads()

  stride = ceil(Int,blockDim().x / 2)
  for i in 1:max_itr
    #一番小さいstrideは1なので、条件に=も含める。
    if tid <= stride
      #@cuprintf("%ld\n",index)
      input[tid + blockIdx_x_Dim] += input[tid + blockIdx_x_Dim + stride]
    end

    CUDAnative.sync_threads()

    #shift演算:一桁ビットをずらすごとに値が半分になる。
    stride >>= 1
  end

  if tid == 1
    output[blockIdx().x] = input[idx]
    #@cuprintf("output[%ld] = %ld\n",blockIdx().x,output[blockIdx().x])
  end

  return nothing
end


#benchmark用にwrapした関数
function bench_reduceNeighbored(input,output,max_itr,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks reduceNeighbored(input,output,max_itr)
    end
end

function bench_reduceNeighbored2(input,output,max_itr,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks reduceNeighbored2(input,output,max_itr)
    end
end

function bench_reduceInterleaved(input,output,max_itr,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks reduceInterleaved(input,output,max_itr)
    end
end

function bench_reduceInterleaved_unroll(input,output,max_itr,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks reduceInterleaved_unroll(input,output,max_itr)
    end
end

function bench_reduceInterleaved_unroll4(input,output,max_itr,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks reduceInterleaved_unroll4(input,output,max_itr)
    end
end

function bench_reduceInterleaved_unroll8(input,output,max_itr,setting::Config)
    CuArrays.@sync begin
        @cuda threads=setting.threads blocks=setting.blocks reduceInterleaved_unroll8(input,output,max_itr)
    end
end

N = 2^24
numthreads = 2^9
numblocks = ceil(Int,N/numthreads)

#log2の返り値はFloatなのでIntでcastする。
max_itr = ceil(Int,log2(numthreads))
println("max_itr :$(max_itr)")
d_input = CuArrays.fill(1,N)
println(length(d_input))
#block間で同期が取れないので、block内での和は全てCPUに持ってきて、最後にCPUで足し上げる。
#なので、output用の配列の大きさはblockの数分必要。
half_numblocks = ceil(Int,numblocks/2)
quarter_numblocks = ceil(Int,numblocks/4)
eighth_numblocks = ceil(Int,numblocks/8)
d_output = CuArrays.zeros(Int,quarter_numblocks)
println("threads:$(numthreads) , blocks:$(numblocks)")
#@cuda threads=numthreads blocks=eighth_numblocks reduceInterleaved_unroll8(d_input,d_output,max_itr )

setting = Config(numthreads,quarter_numblocks)
@time bench_reduceInterleaved_unroll4(d_input,d_output,max_itr,setting)
#@time d = CuArrays.sum(d_input)

#println(d_output)
@show (@test sum(d_output) == sum(fill(1,N)))
