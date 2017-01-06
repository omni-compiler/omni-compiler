      SUBMODULE(m_to_be_read) subm
      CONTAINS
        MODULE FUNCTION f(a)
          REAL :: f
          REAL :: a
          f = a + 1
        END FUNCTION f
      END SUBMODULE subm
