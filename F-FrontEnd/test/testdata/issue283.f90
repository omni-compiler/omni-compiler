module mod1
contains
  subroutine sub1(indata, mask_h)
   integer, intent(inout) :: indata(:)
   logical, optional, intent(in) :: mask_h(:)
   where (.not. mask_h) indata = 0
   if (present (mask_h)) where (.not. mask_h) indata = 0
  end subroutine sub1
end module mod1
