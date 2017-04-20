      MODULE private_member
        TYPE :: t
           REAL, PUBLIC :: i
           REAL, PRIVATE :: k
         CONTAINS
           PROCEDURE, PRIVATE, PASS :: p1
        END TYPE t
        INTERFACE
          MODULE FUNCTION func1(para1)
            INTEGER func1
            INTEGER para1
          END FUNCTION
        END INTERFACE
       CONTAINS
        SUBROUTINE p1(v, ret1)
          CLASS(t) :: v
          INTEGER, INTENT(OUT) :: ret1
          ret1 = v%i + v%k
        END SUBROUTINE
      END MODULE private_member
