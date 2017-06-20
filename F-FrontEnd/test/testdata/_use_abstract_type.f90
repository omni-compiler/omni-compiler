  module m_abstract_derived_type
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
  end module m_abstract_derived_type

  program main
    use abstract_derived_type
  end program main
