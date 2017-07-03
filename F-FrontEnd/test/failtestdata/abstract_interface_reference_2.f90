  program main
    ABSTRACT INTERFACE
      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
      END FUNCTION f
      SUBROUTINE sub(a)
        INTEGER :: a
      END SUBROUTINE sub
    END INTERFACE
    INTEGER :: v
    v = f(v)
  end program main
