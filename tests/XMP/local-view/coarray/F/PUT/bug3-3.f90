program bcast
  include 'xmp_coarray_reduction.h'
  integer a(10,20)

  !----------------------- exec
  call co_broadcast(a,1)

  end
