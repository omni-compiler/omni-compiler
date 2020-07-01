subroutine memory_leak(x)
  real x(2)
  real, allocatable:: a(:)[:]
  allocate (a(10000)[*])
  return
end subroutine memory_leak


program leak_check
  real x(2)

  nalloc0=xmpf_coarray_allocated_bytes()
  ngarbg0=xmpf_coarray_garbage_bytes()
  !!write(*,100) this_image(), nalloc0, ngarbg0

  call memory_leak(x)

  nalloc1=xmpf_coarray_allocated_bytes()
  ngarbg1=xmpf_coarray_garbage_bytes() 
  !!write(*,100) this_image(), nalloc1, ngarbg1

  !!----------- check  -------------
  nerr = 0
  if (nalloc0 /= nalloc1) then
     nerr = nerr + 1
     print '("[",i0,"] NG nalloc0=",i0," nalloc1=",i0)', &
          this_image(), nalloc0, nalloc1
  endif
  if (ngarbg0 /= ngarbg1) then
     nerr = nerr + 1
     print '("[",i0,"] NG ngarbg0=",i0," ngarbg1=",i0)', &
          this_image(), ngarbg0, ngarbg1
  endif

  sync all

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
     call exit(1)
  end if

100 format("[",i0,"] allocated:",i0,", garbage:",i0)

end program
