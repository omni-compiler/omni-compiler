MODULE issue575_2
  USE issue575_1, ONLY: t_model, t_model_m

  IMPLICIT NONE
  PRIVATE

  PUBLIC :: t_js, store

  TYPE t_js
    TYPE(t_model_m), POINTER :: models(:)
  END TYPE t_js

  TYPE(t_js) :: store

END MODULE issue575_2
