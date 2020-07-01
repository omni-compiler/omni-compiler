program test
  write(*,*) "aaa"
contains
  integer function func(k)
    func = k
  end function func
end program test

module external_mmm
contains
  integer function func(k)
    func = k
  end function func
end module external_mmm

subroutine sub
  use external_mmm
  i = func(3)
end subroutine sub

subroutine sub2
  integer func
  i = func(3)
end subroutine sub2

subroutine sub3
  i = func(3)
contains
  integer function func(k)
    func = k
  end function func
end subroutine sub3
