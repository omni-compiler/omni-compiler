      MODULE m_nest_dtag
        INTERFACE t
          MODULE PROCEDURE f
        END INTERFACE t
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
      END MODULE m_nest_dtag

      MODULE m
        use m_nest_dtag
        TYPE t
          INTEGER :: v
        END TYPE t
        INTERFACE s
          MODULE PROCEDURE g
        END INTERFACE s
        TYPE(t) :: a
        TYPE(s) :: b
      END MODULE m

      PROGRAM main
        USE m
        REAL :: r

        a = t(1)
        r = t(2.0)
        b = s(1)
        r = s(2.0)

      END PROGRAM main
