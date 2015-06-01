  program times
    include "xmp_coarray.h"

    real, allocatable :: abig(:)[:,:], bbig(:,:)[:]

    nerr=0

    do i=1,100
       allocate (abig(10000)[2,*],bbig(1000,10)[*])
       if (.not.allocated(abig).or..not.allocated(bbig)) then
          nerr=nerr+1
          write(*,*) "NG: allocatation failed"
       end if

       deallocate (abig,bbig)
       if (allocated(abig).or.allocated(bbig)) then
          nerr=nerr+1
          write(*,*) "NG: deallocatation failed"
       end if
    enddo

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

  end program times
