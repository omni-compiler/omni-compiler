MODULE mod_371
  INTERFACE OPERATOR(.add.)
     MODULE PROCEDURE iadd
     MODULE PROCEDURE fadd
  END INTERFACE OPERATOR(.add.)
CONTAINS
  FUNCTION iadd(i,j) RESULT (ret)
    INTEGER,INTENT(IN) :: i,j
    INTEGER :: ret
    ret = i+j
  END FUNCTION iadd
  FUNCTION fadd(x, y) RESULT (ret)
    real, INTENT(IN) :: x, y
    real :: ret
    ret = x + y
  END FUNCTION fadd
END MODULE mod_371

program test

  USE mod_371, OPERATOR(.x.) => OPERATOR(.add.)

  PRINT *, 1.x.2
  PRINT *, 1 .x. 2

  PRINT *, 1..x.2.
  PRINT *, 1. .x.2.

  PRINT *, 1.5.x.2.
  PRINT *, 1.5 .x. 2.

  PRINT *, -16.E-4.x.2.E4
  PRINT *, -16.E-4 .x. 2.E4

END program test
