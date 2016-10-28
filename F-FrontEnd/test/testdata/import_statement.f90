module mod1

  integer, parameter :: real_size = 4

  interface
    subroutine sub1
      import :: real_size
      real(real_size) :: my_real
    end subroutine sub1
  end interface

end module
