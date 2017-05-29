module mod1
contains
  subroutine sub1()
  end subroutine sub1

  real(kind=8) function func1(a)
    integer :: a
    func1 = 10.0_8
  end function func1
end module mod1