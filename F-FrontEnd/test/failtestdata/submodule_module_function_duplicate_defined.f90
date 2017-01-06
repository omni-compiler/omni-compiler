      MODULE m_module_function_duplicate_defined
        INTERFACE
          MODULE FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE
      END MODULE m_module_function_duplicate_defined

      SUBMODULE(m_module_function_duplicate_defined) subm
      CONTAINS
        MODULE FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
          f = a + 1
        END FUNCTION f
      END SUBMODULE subm

      SUBMODULE(m_module_function_duplicate_defined:subm) subsubm
      CONTAINS
        MODULE FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
          f = a + 2
        END FUNCTION f
      END SUBMODULE subsubm

      PROGRAM main
        USE m_module_function_duplicate_defined
      END PROGRAM main
