      MODULE m_dtagim
        INTERFACE t
          MODULE PROCEDURE f
        END INTERFACE t
        TYPE t
          INTEGER :: v
        END TYPE t
        INTERFACE s
          MODULE PROCEDURE g
        END INTERFACE s
        TYPE s
          INTEGER :: v
        END TYPE s
       CONTAINS
        FUNCTION f(a)
          REAL :: f
          REAL :: a
          f = a
        END FUNCTION f
        FUNCTION g(a)
          REAL :: g
          REAL :: a
          g = a
        END FUNCTION g
      END MODULE m_dtagim
