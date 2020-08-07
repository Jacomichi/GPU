using CUDAnative

function say(num)
  @cuprintf("Thread %ld says: %ld\n",threadIdx().x,num)
  return
end

@cuda threads=4 say(42)
