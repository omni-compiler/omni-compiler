  module abstract_derived_type_2
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
     CONTAINS
      PROCEDURE :: p => ftt
    END TYPE

   contains
    FUNCTION ftt(this)
      CLASS(tt) :: this
    END FUNCTION ftt
  end module abstract_derived_type_2

  program main
    use abstract_derived_type_2
  end program main
