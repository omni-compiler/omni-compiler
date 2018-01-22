      PROGRAM test_same_type_as
        TYPE t
        END TYPE t
        TYPE,EXTENDS(t) :: tt
        END type tt

        TYPE(t) :: a
        TYPE(tt) :: b
        PRINT *, SAME_TYPE_AS(a, b)
      END PROGRAM test_same_type_as
