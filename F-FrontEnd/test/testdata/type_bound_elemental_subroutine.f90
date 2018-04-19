      MODULE m
        TYPE t
          INTEGER :: v
         CONTAINS
          PROCEDURE,PASS :: sub
        END TYPE
       CONTAINS
        ELEMENTAL SUBROUTINE sub(a)
          CLASS(t),INTENT(IN) :: a
        END SUBROUTINE sub
      END MODULE m

      PROGRAM main
        USE m
        TYPE(t), DIMENSION(1:3) :: v
        CALL v%sub()
      END PROGRAM main
