  real,allocatable:: ad0(:,:)[:]

  allocate(ad0(1:0,1:10)[*])   !! size zero
  call nfoo(ad0)

  nerr=0
  if (allocated(ad0)) nerr=nerr+1

  if (nerr==0) then
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if

  contains

    subroutine nfoo(d0)
      real, allocatable :: d0(:,:)[:]

      deallocate(d0)

      return
    end subroutine nfoo
  end
