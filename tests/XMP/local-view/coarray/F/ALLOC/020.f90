MODULE real_poly_module
  TYPE,PUBLIC :: real_poly
    REAL,ALLOCATABLE :: coeff(:)
  END TYPE
END MODULE

PROGRAM example
  USE real_poly_module
  TYPE(real_poly) poly1
  poly1 = real_poly((/1.0,2.0,4.0/))
  IF (.NOT.ALLOCATED(poly1%coeff)) STOP 'NOT ALLOCATED'
  PRINT *,'PASS'
END

