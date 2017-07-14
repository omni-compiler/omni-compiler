  module m
    TYPE :: t
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
  end module m
