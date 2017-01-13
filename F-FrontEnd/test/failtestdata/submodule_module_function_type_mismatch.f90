      MODULE m_module_function_type_mismatch
        INTERFACE
           MODULE FUNCTION f(a)
             REAL :: f
             REAL :: a
           END FUNCTION f
        END INTERFACE
      END MODULE m_module_function_type_mismatch

      SUBMODULE(m_module_function_type_mismatch) subm
      CONTAINS
        MODULE FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
        END FUNCTION f
      END SUBMODULE subm

      PROGRAM main
        USE m_module_function_type_mismatch
      END PROGRAM main
