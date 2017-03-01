      MODULE m_separate_module_procedure_duplicate_defined
        INTERFACE
          MODULE FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE
      END MODULE m_separate_module_procedure_duplicate_defined

      SUBMODULE(m_separate_module_procedure_duplicate_defined) subm
      CONTAINS
        MODULE PROCEDURE f
          f = a + 1
        END PROCEDURE f
      END SUBMODULE subm

      SUBMODULE(m_separate_module_procedure_duplicate_defined:subm) subsubm
      CONTAINS
        MODULE PROCEDURE f
          f = a + 2
        END PROCEDURE f
      END SUBMODULE subsubm

      PROGRAM main
        USE m_separate_module_procedure_duplicate_defined
      END PROGRAM main
