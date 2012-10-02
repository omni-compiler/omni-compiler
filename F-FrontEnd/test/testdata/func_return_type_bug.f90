! this test program is for reproduction of a bug that function which returns
! real array is compiled to 'FnumericAll' array in XcodeML which F_Back ignores
module mod_func_return_type_bug
  implicit none
  private
contains
  function func( ijdim ) 
    implicit none
    integer, intent(in) :: ijdim
    real(8)             :: func(ijdim)
  end function func
end module mod_func_return_type_bug
!-------------------------------------------------------------------------------
