MODULE mod1

IMPLICIT NONE

CONTAINS
  SUBROUTINE sub1(values, mask_a, mask_b)
    real, intent(inout) :: values(:)
    logical, optional ,intent(in) :: mask_a (:)
    logical, optional ,intent(in) :: mask_b (:)

    if (present(mask_a)) values(:) = 0.5
  
    if (present(mask_a)) then 
      where(.not. mask_a) values = 0.0
    end if
    if (present(mask_b)) where (.not. mask_b) values = 1.0
  END SUBROUTINE sub1
END MODULE mod1
