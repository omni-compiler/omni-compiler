      program main
        integer a 
        a = 1
!$      a = 2    ! condition compilation without no -no-omp option.
        print *,a
      end program main

