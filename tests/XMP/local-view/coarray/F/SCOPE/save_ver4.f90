!! for --Wx-fcoarray=4 option

  program main
    call foo
  end program main

  subroutine foo
!!     include 'xmp_coarray.h'
!!    save
    integer, save :: sss[*]
    integer :: ttt[*]

  end subroutine foo
