module mod1
contains
  subroutine sub1()
  end subroutine

  function fct1()
    real :: fct1
  end function

  subroutine sub2(sub1, fct1)
    external sub1, fct1
  end subroutine sub2

  subroutine sub3()
    call sub2(sub1, fct1)
  end subroutine sub3
end module mod1
