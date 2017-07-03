      PROGRAM main
        INTEGER, DIMENSION(:), ALLOCATABLE :: a
        INTEGER, DIMENSION(8) :: b

        ALLOCATE(a, source=b)

        ALLOCATE(a, source=b(3:4))

      END PROGRAM main
