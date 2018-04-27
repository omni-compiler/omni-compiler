MODULE issueXX_mod

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: t_action, t_hsm

   TYPE t_Action
     CHARACTER(len=10) :: name = ""
   CONTAINS
   END TYPE t_Action
 
   INTERFACE t_Action
     PROCEDURE Construct_action
   END INTERFACE t_Action

   TYPE :: t_Hsm
     CHARACTER(len=30)           :: name
     !
     TYPE(t_Action),   ALLOCATABLE :: actions(:)         !< List of possible actions (events)
   CONTAINS
     PROCEDURE :: Register_action   => Register_action_hsm
   END TYPE t_Hsm

CONTAINS

   FUNCTION Construct_action(name) RESULT(return_ptr)
     CHARACTER(len=*), INTENT(in) :: name
     TYPE(t_Action), POINTER :: return_ptr
     ALLOCATE(return_ptr)
     return_ptr%name = TRIM(name)
   END FUNCTION Construct_action

   SUBROUTINE Register_action_hsm(this, action, debug)
 
     CLASS(t_Hsm),   INTENT(inout) :: this
     TYPE(t_Action), INTENT(in)    :: action
     LOGICAL, OPTIONAL, INTENT(in) :: debug
 
   END SUBROUTINE Register_action_hsm


END MODULE issuexx_mod
