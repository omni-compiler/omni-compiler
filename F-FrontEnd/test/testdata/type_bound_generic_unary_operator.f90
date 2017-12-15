      MODULE m
        TYPE t
         INTEGER :: i
         CONTAINS
          PROCEDURE :: f
          GENERIC :: OPERATOR(-) => f
        END TYPE
      CONTAINS
        FUNCTION f(a)
         CLASS(t),INTENT(IN) :: a
         TYPE(t) :: f
         f%i = - a%i
        END FUNCTION
      END MODULE
