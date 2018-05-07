MODULE mod1
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: t_state

  TYPE :: t_state
    INTEGER :: i
    REAL :: r
  END TYPE t_State

  INTERFACE t_state
    PROCEDURE construct_state
  END INTERFACE t_state

CONTAINS

  FUNCTION construct_state(i, r) RESULT(return_ptr)
    INTEGER, INTENT(IN) :: i
    REAL, INTENT(IN) :: r
    TYPE(t_state), POINTER :: return_ptr

    ALLOCATE(return_ptr)
    return_ptr%i = i
    return_ptr%r = r
  END FUNCTION construct_state

  SUBROUTINE init_state(this)
    CLASS(t_state), POINTER :: this 

    this => t_state(1, 1.0)
  END SUBROUTINE init_state
END MODULE mod1
