module mod1

contains

  subroutine sub1(array)
    integer, allocatable, intent(inout) :: array(:)

    allocate(array(10))
  end subroutine sub1

  subroutine sub2(array)
    integer, intent(in) :: array(:)

  end subroutine sub2

  subroutine sub3(array)
    integer, allocatable, intent(out) :: array(:)

    allocate(array(10))
  end subroutine sub3

end module mod1
