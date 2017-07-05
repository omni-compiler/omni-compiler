      PROGRAM main
      CONTAINS
        SUBROUTINE sub()
          INTEGER :: i
          INTEGER :: v(0:1)
          FORALL(i = 0:1)
            v(i) = 0
          END FORALL
        END SUBROUTINE sub
      END PROGRAM main
