program test

  interface
     subroutine sub(a)
       real a(:)
     end subroutine sub
  end interface

  type tt
     real, allocatable :: x(:)
  end type tt

  type ss
     type(tt) y(10)
  end type ss

  type(ss) z(100)

  allocate (z(1)%y(1)%x(2:100))

  call sub(z(1)%y(1)%x(2:100))

end program test
