program bcast
!!   include 'xmp_coarray.h'
  integer a(10,20)

  !----------------------- exec
  call co_broadcast(a,1)

  end
