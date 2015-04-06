program allo2
  include "xmp_lib.h"
  real, allocatable :: aaa(:)[:,:] 

  write(*,*) "f", allocated(aaa)
  call allo
  write(*,*) "t", allocated(aaa)
  write(*,*) "size(aaa)=10", size(aaa)
  call reallo
  write(*,*) "t", allocated(aaa)
  write(*,*) "size(aaa)=100", size(aaa)
  stop

contains
  subroutine allo
    write(*,*) "f", allocated(aaa)
    allocate (aaa(10)[2,*])
    write(*,*) "t", allocated(aaa)
  end subroutine allo
      
  subroutine reallo
    deallocate (aaa)
    write(*,*) "f", allocated(aaa)
    allocate (aaa(100)[4,*])
  end subroutine reallo
      
end program allo2
