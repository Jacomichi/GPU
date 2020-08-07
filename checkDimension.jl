using CUDAnative

function checkIndex()
  @cuprintf("threadIdx:(%ld,%ld,%ld)\n",threadIdx().x,threadIdx().y,threadIdx().z)
  @cuprintf("blockIdx:(%ld,%ld,%ld)\n",blockIdx().x,blockIdx().y,blockIdx().z)
  @cuprintf("blockDim:(%ld,%ld,%ld)\n",blockDim().x,blockDim().y,blockDim().z)
  @cuprintf("gridDim:(%ld,%ld,%ld)\n",gridDim().x,gridDim().y,gridDim().z)
  return
end

nElem = 6
nthreads = (2,2)
nblocks = (1,1)

@cuda threads=nthreads blocks=nblocks checkIndex()
#0 println("")
#@cuda threads=nElem checkIndex()

#@cuprintf("threadIdx:(%d,%d,%d)blockIdx:(%d,%d,%d)blockDim:(%d,%d,%d)gridDim:(%d,%d,%d)\n",
#threadIdx().x,threadIdx().y,threadIdx().z,blockIdx().x,blockIdx().y,blockIdx().z,
#blockDim().x,blockDim().y,blockDim().z,gridDim().x,gridDim().y,gridDim().z)
