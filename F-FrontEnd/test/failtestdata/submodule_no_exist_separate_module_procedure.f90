      MODULE m_no_exist_module_procedure
        INTERFACE
          MODULE FUNCTION g(a)
            INTEGER :: g
            INTEGER :: a
          END FUNCTION g
        END INTERFACE
      END MODULE m_no_exist_module_procedure

      SUBMODULE(m_no_exist_module_procedure) subm
      CONTAINS
        MODULE PROCEDURE f
          f = a + 1
        END PROCEDURE f
      END SUBMODULE subm

      PROGRAM main
        USE m_no_exist_module_procedure
      END PROGRAM main
