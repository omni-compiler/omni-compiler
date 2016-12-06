      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, PUBLIC :: f
        END TYPE t
      CONTAINS
        FUNCTION f(i, j)
          INTEGER :: f
          INTEGER :: i
          !CLASS(t) :: i
          INTEGER :: j
          PRINT *, "call F"
        END FUNCTION f
      END MODULE m

      PROGRAM main
        USE m
      END PROGRAM main
