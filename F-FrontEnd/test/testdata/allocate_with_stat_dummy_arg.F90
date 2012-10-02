      subroutine sub(t)
      integer, intent(inout) :: t(2)
      integer, allocatable :: a(:)

      allocate(a(10), stat = t(1))

      end subroutine sub

      program main
      integer :: t(2)

      call sub(t)

      end program main

