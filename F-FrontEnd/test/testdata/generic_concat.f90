      MODULE m
        TYPE t
          CHARACTER(3) :: v = 'abc'
         CONTAINS
          PROCEDURE :: f
          GENERIC :: OPERATOR(//) => f
         END TYPE
        CONTAINS
         ELEMENTAL FUNCTION f(a, b)
           CLASS(t), INTENT(IN) :: a
           CHARACTER(10), INTENT(IN) :: b
           CHARACTER(13) :: f
           f = a%v // b
         END FUNCTION f
      END MODULE m

      PROGRAM main
        USE m
        TYPE(t),DIMENSION(1:3), TARGET :: a
        CHARACTER(10) :: c = 'efg'
        PRINT *, a // c
      END PROGRAM main
