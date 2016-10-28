      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, NOPASS, PUBLIC :: f
           PROCEDURE, NOPASS, PUBLIC :: g
           PROCEDURE, NOPASS, PUBLIC :: h
           GENERIC :: p => f, g, h
        END TYPE t
      CONTAINS
        SUBROUTINE f(i)
          INTEGER(KIND=4) :: i
          PRINT *, "call F"
        END SUBROUTINE f
        SUBROUTINE g(r)
          INTEGER(KIND=8) :: r
          PRINT *, "call G"
        END SUBROUTINE g
        SUBROUTINE h(r)
          INTEGER(KIND=16) :: r
          PRINT *, "call H"
        END SUBROUTINE h
      END MODULE m

      PROGRAM main
        USE m
        TYPE(t), TARGET :: a
        CLASS(t), POINTER :: b
        b => a
        CALL b%p(1_4)
        CALL b%p(1_8)
        CALL b%p(1_16)
        !CALL b%p(1_32)
      END PROGRAM main
