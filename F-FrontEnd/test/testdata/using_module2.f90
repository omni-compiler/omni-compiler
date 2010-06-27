      module m0
        implicit none
        type tt
            integer a
        end type tt
      end module m0

      program main
        use m0
        implicit none
        type(tt) :: c
      end program main
