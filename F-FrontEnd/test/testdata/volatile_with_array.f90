      PROGRAM main
        INTEGER :: i(2)
       CONTAINS
        SUBROUTINE s
          VOLATILE i
          i(1) = 3
        END SUBROUTINE s
      END PROGRAM main
