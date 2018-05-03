MODULE mod1

  USE issue575_3, ONLY: store
  USE issue575_1, ONLY: t_model

  IMPLICIT NONE
  PRIVATE

CONTAINS

  SUBROUTINE sub1(model_id)
    INTEGER, INTENT(IN) :: model_id
    TYPE(t_model),  POINTER  :: model

    model => store%models(model_id)%m
  END SUBROUTINE sub1

END MODULE mod1
