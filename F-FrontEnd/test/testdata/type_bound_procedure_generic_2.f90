      MODULE m
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE, NOPASS, PUBLIC :: f
           PROCEDURE, NOPASS, PUBLIC :: g
           GENERIC :: p => f, g
        END TYPE t
        TYPE, EXTENDS(t) :: tt
         CONTAINS
           PROCEDURE, NOPASS, PUBLIC :: g => h
        END TYPE tt
      CONTAINS
        SUBROUTINE f(i)
          INTEGER :: i
          PRINT *, "call F"
        END SUBROUTINE f
        SUBROUTINE g(r)
          REAL :: r
          PRINT *, "call G"
        END SUBROUTINE g
        SUBROUTINE h(r)
          REAL :: r
          PRINT *, "call H"
        END SUBROUTINE h
      END MODULE m

      PROGRAM main
        USE m
        TYPE(t), TARGET :: a
        CLASS(t), POINTER :: b
        TYPE(tt), TARGET :: c
        b => a
        CALL b%p(1)
        CALL b%p(1.2)
        b => c
        CALL b%p(1)
        CALL b%p(1.2)
      END PROGRAM main
