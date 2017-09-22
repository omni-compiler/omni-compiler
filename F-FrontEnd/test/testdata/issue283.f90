module mod1

  integer, parameter :: wp = selected_real_kind(13)

contains


  subroutine get_var_int (name, var, ierr, fill)
   character(len=*)   ,intent(in)  :: name    ! variable name
   integer            ,intent(out) :: var (:) ! variable returned
   integer  ,optional ,intent(out) :: ierr    ! error return parameter
   integer  ,optional ,intent(in)  :: fill    ! fillvalue to use
  end subroutine get_var_int

  subroutine get_var_real (name, var, ierr, fill)
    character(len=*)   ,intent(in)  :: name    ! variable name
    real(wp)           ,intent(out) :: var (:) ! variable returned
    integer  ,optional ,intent(out) :: ierr    ! error return parameter
    real(wp) ,optional ,intent(in)  :: fill    ! fillvalue to use

  end subroutine get_var_real

  subroutine sub1()
    integer, dimension(10) :: v1
    real(wp), dimension(10) :: v2

    call get_var_int  ('dbkz', v1, fill=-1)
    call get_var_real ('sat_', v2, fill=-999._wp)
  end subroutine sub1

end module mod1
