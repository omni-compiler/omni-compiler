  !$xmp nodes p(8)
  !$xmp nodes q1(3)=p(1:3)
  !$xmp nodes q2(2,2)=p(4:7)

  real as(10,10)[*]
  real,allocatable:: ad0(:,:)[:],ad1(:,:)[:]

  if (xmpf_coarray_uses_fjrdma()) then
     if (this_image()==1) then
        write(*,*) "skip autodealloc1.f90 because FJRDMA causes SIGSEGV"
        write(*,*) "  SEGV_MAPERR, address not mapped to object"
     endif
     stop
  endif

  allocate(ad0(1:0,1:10)[*])   !! size zero
  allocate(ad1(1:1,1:10)[*])
  call nfoo(as,ad0,ad1)
!!  write(*,*) "allocated_bytes= ", xmpf_coarray_allocated_bytes()
!!  write(*,*) "garbage_bytes  = ", xmpf_coarray_garbage_bytes()

  nerr=0
  if (.not.allocated(ad0)) nerr=nerr+1
  if (allocated(ad1))      nerr=nerr+1

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
     call exit(1)
  end if



  contains

    subroutine nfoo(s,d0,d1)
      real s(10,10)
      real, allocatable :: d0(:,:)[:],d1(:,:)[:]

      deallocate(d1)

      return
    end subroutine nfoo

  end 


