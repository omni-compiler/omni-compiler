  PROGRAM main
    TYPE :: t
      INTEGER :: v
      INTEGER, ALLOCATABLE, DIMENSION(:) :: w
    END TYPE

    TYPE(t) :: a

    a = t(1)
  END PROGRAM main
