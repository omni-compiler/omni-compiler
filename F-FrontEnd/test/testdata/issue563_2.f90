MODULE mo_jsb_model_class

  USE issuexx_mod, ONLY: t_Hsm, t_action

  IMPLICIT NONE

CONTAINS

  FUNCTION new_model() RESULT(return_ptr)
    TYPE(t_hsm), POINTER             :: return_ptr

    ALLOCATE(return_ptr)

    CALL return_ptr%Register_action(t_Action("INTEGRATE"), debug=.FALSE.)
    CALL return_ptr%Register_action(t_Action("AGGREGATE"), debug=.FALSE.)

  END FUNCTION new_model
END MODULE mo_jsb_model_class
