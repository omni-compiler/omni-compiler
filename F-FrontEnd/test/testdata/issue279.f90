MODULE mod1
CONTAINS

  SUBROUTINE sub1(param1)
    REAL(KIND=8), INTENT(OUT) :: param1
  
    param1 = 8.0
  END SUBROUTINE sub1

  FUNCTION param1()
    REAL (KIND=8) :: param1
 
    param1  =  8.0
  END FUNCTION param1
END MODULE mod1
