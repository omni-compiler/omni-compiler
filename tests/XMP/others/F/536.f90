program test
  TYPE t
     INTEGER :: v(8)
  END TYPE t
  TYPE(t) :: u(5)
  PRINT *, u(1:5)%v(1) !=> the array with 5 integer
END program test
