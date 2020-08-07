using CUDAnative

function say(f)
  i = threadIdx().x
  @cuprintf("Thread %ld says: %ld\n",i,f(i))
  return
end

@cuda threads=4 say(x->x+1)
