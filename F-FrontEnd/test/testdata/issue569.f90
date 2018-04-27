MODULE mod1

  IMPLICIT NONE

  TYPE :: t_var
    REAL, ALLOCATABLE :: field(:,:,:,:,:)
  END TYPE t_var

  TYPE, EXTENDS(t_var) :: t_real2d
    REAL, POINTER :: ptr(:,:) => NULL()
  END TYPE t_real2d

CONTAINS

  SUBROUTINE sub1(var)
    TYPE(t_real2d), TARGET, INTENT(inout) :: var 
    var%ptr => var%field(:,:,1,1,1)
  END SUBROUTINE sub1
END MODULE mod1
