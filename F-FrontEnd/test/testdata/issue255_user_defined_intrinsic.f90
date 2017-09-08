module mod1

  contains
  
  subroutine random_number(a, b)
    integer, intent(in) :: a, b 
  end subroutine random_number

  subroutine sub1()
    call random_number(1, 2)
  end subroutine sub1

end module mod1
