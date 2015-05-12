  program main
    include "xmp_coarray.h"
    real,save :: ms(10)[*]
    real m(10)[*]
    real,allocatable :: ma(:)[:]
    real,allocatable,save:: msa(:)[:]

    integer dondon
    do i=1,10
       write(*,*) dondon(i)
    enddo
  end program main

  recursive function dondon(z) result(ans)
    include "xmp_coarray.h"
    integer z, ans
    real,save :: ps(10)[*]
    real p(10)[*]                       !! only this var is error
    real,allocatable :: pa(:)[:]
    real,allocatable,save:: psa(:)[:]

    if (z==0) then
       ans=1
       return
    endif

    ans=dondon(z-1)*z
    return
  end function dondon

