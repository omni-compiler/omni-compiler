MODULE mod1

  IMPLICIT NONE

  TYPE, ABSTRACT :: t_var
    REAL, ALLOCATABLE :: field(:,:,:,:,:)
  END TYPE t_var

  TYPE, EXTENDS(t_var) :: t_real2d
    REAL, POINTER :: ptr(:,:) => NULL()
  END TYPE t_real2d

  TYPE t_var_p
    CLASS(t_var), POINTER :: p
  END TYPE t_var_p

  TYPE, ABSTRACT :: t_memory
    TYPE(t_var_p), ALLOCATABLE :: vars(:)
  END TYPE t_memory

CONTAINS

  SUBROUTINE sub1(mem, var)

    CLASS(t_memory), INTENT(inout)        :: mem    
    TYPE(t_real2d), TARGET, INTENT(inout) :: var 
    INTEGER :: pos_of_var

    mem%vars(pos_of_var)%p => var

  END SUBROUTINE sub1

END MODULE mod1
