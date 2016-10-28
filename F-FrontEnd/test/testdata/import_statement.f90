module mod1
  integer, parameter :: real_size = 8 
  integer, parameter :: integer_size = 4

  interface
    subroutine sub1
      import :: real_size
      real(real_size) :: my_real
    end subroutine sub1

    subroutine sub2
      import :: real_size, integer_size
      real(real_size) :: my_real
      integer(integer_size) :: my_int
    end subroutine sub2
      
  end interface

end module
