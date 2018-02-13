SUBROUTINE allocate
  INTEGER, ALLOCATABLE :: a
  DIMENSION :: a(:)
  allocate (a(2))
END SUBROUTINE allocate
