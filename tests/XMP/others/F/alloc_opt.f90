program main
  type ttt
     real x
  end type ttt
  type(ttt) a
  type(ttt), allocatable :: b, c
  character s
  allocate(b, c, source=a, stat=k, errmsg=s)
  deallocate(b, c, stat=k, errmsg=s)
end program main
