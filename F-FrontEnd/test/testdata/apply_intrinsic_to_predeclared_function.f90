MODULE mod1
  CONTAINS

  SUBROUTINE sub()
    REAL, POINTER :: z(:,:) => NULL()     
    z(:,:) = RESHAPE(f(), (/2, 2/))
  END SUBROUTINE sub

  FUNCTION f()
    IMPLICIT NONE
    REAL, DIMENSION(4) :: f
  END FUNCTION f

END MODULE mod1

PROGRAM main; END PROGRAM main
