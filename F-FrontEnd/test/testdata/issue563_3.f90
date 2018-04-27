MODULE mod1
  USE issue563_mod, ONLY: t_msg
  USE issue563_3_mod, ONLY: t_action, t_hsm


  IMPLICIT NONE
  PRIVATE
  PUBLIC :: t_jsb_model


  TYPE, EXTENDS(t_Hsm) :: t_jsb_model

  END TYPE t_jsb_model

CONTAINS



  SUBROUTINE sub1(this)
    CLASS(t_jsb_model), INTENT(inout) :: this
    CLASS(t_msg), POINTER :: msg
 
    msg => t_msg('', this%Get_action('INTEGRATE'))
  END SUBROUTINE sub1

END MODULE mod1
