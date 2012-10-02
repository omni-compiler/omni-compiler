      PROGRAM main
        USE private_struct
        IMPLICIT NONE
        INTEGER,PARAMETER :: i = p%n
        TYPE(ttt) :: w
        v%n = 1
        v = p
        v = f()

        w = u
        w%p%n = 1
        w%p = p
      END PROGRAM main

