  program decl
    include "xmp_coarray.h"
    real, allocatable :: a(:)[:]
    integer, allocatable :: s[:]
    integer nerr = 0
    integer me

    me = this_image()

    if (allocated(a)) then
       nerr = nerr+1
       write(*,100) me, "allocated(a) sould be false"
    endif

!!! restriction due to OMNI   
!!    if (allocated(s)) then
!!       nerr = nerr+1
!!       write(*,100) me, "allocated(s) sould be false"
!!    endif

    if (nerr==0) then
       write(*,100) me, "OK"
    else
       write(*,101) me, "NG", nerr
    endif

100 format("[",i0,"] ",a) 
101 format("[",i0,"] ",a," nerr=",i0) 

  end program decl
