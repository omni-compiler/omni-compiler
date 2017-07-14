  program main
    ABSTRACT INTERFACE
      FUNCTION f(a)
        INTEGER :: f
        INTEGER :: a
      END FUNCTION f
    END INTERFACE
    INTEGER :: v
    PROCEDURE(f), POINTER :: p
    p => f
  end program main
