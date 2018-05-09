      PROGRAM test_extends_type_of
        TYPE t
        END TYPE t
        TYPE,EXTENDS(t) :: tt
        END type tt

        TYPE(t) :: a
        TYPE(tt) :: b
        PRINT *, EXTENDS_TYPE_OF(b, a)
      END PROGRAM test_extends_type_of
