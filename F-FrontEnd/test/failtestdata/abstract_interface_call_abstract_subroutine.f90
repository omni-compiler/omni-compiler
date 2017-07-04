  program main
    ABSTRACT INTERFACE
      SUBROUTINE sub(a)
        INTEGER :: a
      END SUBROUTINE sub
    END INTERFACE
    CALL sub(1)
  end program main
