      MODULE mod
        TYPE matrix
           REAL*4 element(100,100)
        END TYPE matrix
      END MODULE mod

      PROGRAM main
        USE mod
        TYPE(matrix) :: a
        TYPE(matrix),ALLOCATABLE :: b,c,d,e

        ALLOCATE(b,source=a)
        ALLOCATE(c,source=a)

        !ALLOCATE(matrix::d,e)
      END PROGRAM main
