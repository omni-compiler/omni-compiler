      PROGRAM main
        INTEGER, DIMENSION(:), ALLOCATABLE :: a
        INTEGER, DIMENSION(8) :: b

        ALLOCATE(a, mold=b)

        ALLOCATE(a, mold=b(3:4))

      END PROGRAM main
