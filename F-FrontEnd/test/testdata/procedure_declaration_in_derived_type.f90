      PROGRAM main
        TYPE t
          INTEGER :: v = 0
          PROCEDURE(f), POINTER, NOPASS :: u => null()
          PROCEDURE(h), POINTER, PASS :: w
        END TYPE t
        INTERFACE
          FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE
        INTERFACE
          FUNCTION h(a)
            IMPORT t
            INTEGER :: h
            CLASS(t) :: a
          END FUNCTION h
        END INTERFACE
        TYPE(t) :: v
        v%u => g
        v%w => h
        v%w => i
        v%v = v%u(1)
        PRINT *, v%v
      CONTAINS
        FUNCTION g(a)
          INTEGER :: g
          INTEGER :: a
          g = a + 1
        END FUNCTION g
        FUNCTION i(a)
          INTEGER :: i
          CLASS(t) :: a
        END FUNCTION i
      END PROGRAM main
