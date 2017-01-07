      MODULE m_separate_module_procedure
        INTERFACE
           MODULE FUNCTION f(a)
             INTEGER(KIND=8) :: f
             INTEGER(KIND=8) :: a
           END FUNCTION f
           MODULE SUBROUTINE g(a)
             REAL(KIND=8) :: a
           END SUBROUTINE g
        END INTERFACE
      END MODULE m_separate_module_procedure

      SUBMODULE(m_separate_module_procedure) subm
      CONTAINS
        MODULE PROCEDURE f
          f = a + 1
        END PROCEDURE f
        MODULE PROCEDURE g
          PRINT *, a
        END PROCEDURE g
      END SUBMODULE subm

      PROGRAM main
        USE m_separate_module_procedure
      END PROGRAM MAIN
