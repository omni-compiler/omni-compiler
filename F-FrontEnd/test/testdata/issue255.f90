module mod1
  use selected_kind

  type typ1
    real(dp) :: rinvalid = -huge(1._dp)
  end type typ1

end module mod1
