  module m
    TYPE, ABSTRACT :: t
      INTEGER :: v
     CONTAINS
      PROCEDURE(f), DEFERRED :: p
    END TYPE
    ABSTRACT INTERFACE
      FUNCTION f(this)
        IMPORT t
        CLASS(t) :: this
      END FUNCTION f
    END INTERFACE
    TYPE, EXTENDS(t) :: tt
    END TYPE
  end module m
