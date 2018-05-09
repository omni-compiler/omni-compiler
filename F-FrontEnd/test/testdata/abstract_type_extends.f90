  module m
    TYPE, ABSTRACT :: t
      INTEGER :: v
     CONTAINS
      PROCEDURE(f), DEFERRED :: p
    END TYPE

    TYPE, ABSTRACT, EXTENDS(t) :: tt
    END type tt

    INTERFACE
      FUNCTION f(this)
        IMPORT t
        CLASS(t) :: this
        INTEGER :: f
      END FUNCTION f
    END INTERFACE

  end module m

  program main
    use m
  end program main
