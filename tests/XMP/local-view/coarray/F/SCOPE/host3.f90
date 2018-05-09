  program main
    call sub1
  end program main

  subroutine sub1
    real a(10)[*]

  contains
    subroutine sub2
      integer a(10)[*]

    end subroutine sub2
  end subroutine sub1


!! $ xmpf90 host3.f90 
!! host3.f90:11.25:
!! 
!!   INTEGER :: a ( 1 : 10 )
!!                          1
!! Error: Duplicate array spec for Cray pointee at (1)
!! $ 
