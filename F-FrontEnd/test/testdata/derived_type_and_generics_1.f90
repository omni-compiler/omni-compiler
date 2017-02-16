      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
        f = a
      END FUNCTION f

      FUNCTION g(a)
        INTEGER :: g
        INTEGER :: b
        g = a
      END FUNCTION g

      PROGRAM main
        INTERFACE t
          FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE t
        TYPE t
          REAL :: v
        END TYPE t

        TYPE s
          REAL :: v
        END TYPE s
        INTERFACE s
          FUNCTION g(a)
            INTEGER :: g
            INTEGER :: a
          END FUNCTION g
        END INTERFACE s

        TYPE(t) :: a
        TYPE(s) :: b

        a = t(1.0)
        a%v = t(2)

        b = s(1.0)
        b%v = s(2)

      END PROGRAM main
