  real as(10,10)[*]
  real,allocatable:: ad(:,:)[:]
  
  nerr=0
  if (allocated(ad)) then
     nerr=nerr+1
  endif
  m=nfoo(as,ad)
  if (m.ne.33) then
     nerr=nerr+1
  endif
  if (.not.allocated(ad)) then
     nerr=nerr+1
  endif
  do j=1,10
     do i=1,10
        if (i==3.and.j==5) then
           if (ad(i,j).ne.2.0) nerr=nerr+1
        else
           if (ad(i,j).ne.1.0) nerr=nerr+1
        endif
     enddo
  enddo

  write(*,*) "allocated_bytes= ", xmpf_coarray_allocated_bytes()  !! should be 800!
  write(*,*) "garbage_bytes  = ", xmpf_coarray_garbage_bytes()

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if

  contains

    integer function nfoo(s,d)
      real s(10,10)[*]
      real, allocatable :: d(:,:)[:]

      allocate (d(10,10)[*])
      write(*,*) "(in) allocated_bytes= ", xmpf_coarray_allocated_bytes()  !! should be 800
      write(*,*) "(in) garbage_bytes  = ", xmpf_coarray_garbage_bytes()


      nfoo = 33
      d = 1.0
      d(3,5)=2.0
      return
    end subroutine nfoo

  end
