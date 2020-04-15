subroutine sub
  call intersub
  return
contains
  subroutine intersub
    real,allocatable:: c(:)[:],d(:,:)[:],e(:)[:]
    allocate(c(50)[*], d(20,4)[*], e(60)[*])
    deallocate(c,e)
    return
  end subroutine intersub
end subroutine sub


program main
  nalloc0=xmpf_coarray_allocated_bytes()
  ngarbg0=xmpf_coarray_garbage_bytes()
  !!write(*,100) this_image(), nalloc0, ngarbg0
  call sub
  nalloc1=xmpf_coarray_allocated_bytes()
  ngarbg1=xmpf_coarray_garbage_bytes() 
  !!write(*,100) this_image(), nalloc1, ngarbg1
100 format("[",i0,"] allocated:",i0,", garbage:",i0)

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

end program
