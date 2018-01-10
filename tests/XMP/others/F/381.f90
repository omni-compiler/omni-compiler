program test
  TYPE t
     INTEGER, DIMENSION(:), POINTER::v
  END TYPE t

  TYPE(t), ALLOCATABLE, DIMENSION(:) :: a

  ALLOCATE(a(1)%v(2))

END program test
