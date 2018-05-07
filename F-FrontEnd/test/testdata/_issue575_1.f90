MODULE issue575_1
  IMPLICIT NONE
  PRIVATE

  PUBLIC :: t_model_m, t_model, t_hsm
 
  TYPE, ABSTRACT :: t_hsm
  END TYPE t_hsm

  TYPE, EXTENDS(t_hsm) :: t_model
  END TYPE t_model

  TYPE t_model_m
    TYPE(t_model), POINTER :: m
  END TYPE t_model_m

END MODULE issue575_1
