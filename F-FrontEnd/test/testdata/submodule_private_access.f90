      MODULE m_private_access
        TYPE :: t
           REAL, PUBLIC :: i
           REAL, PRIVATE :: k
         CONTAINS
           PROCEDURE, PRIVATE :: p
        END TYPE t
        COMPLEX :: i
        PRIVATE :: i, t
        INTERFACE
          MODULE SUBROUTINE sub()
          END SUBROUTINE sub
        END INTERFACE
      CONTAINS
        SUBROUTINE p(v)
          CLASS(t) :: v
          PRINT *, v%k
        END SUBROUTINE p
      END MODULE m_private_access

      SUBMODULE(m_private_access) subm
      CONTAINS
        MODULE SUBROUTINE sub()
          TYPE(t), POINTER :: v
          TYPE(t), TARGET :: u
          COMPLEX :: r = (1,1)
          v => u
          v%i = REAL(r)
          v%k = IMAG(r)
          CALL v%p()
        END SUBROUTINE sub
      END SUBMODULE subm

      PROGRAM main
        USE m_private_access
        CALL sub()
      END PROGRAM main
