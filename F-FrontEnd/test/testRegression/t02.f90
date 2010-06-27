      program  array_sum
      implicit none

      real(8) :: a(10)

      real sum
      external sum

      integer:: i
      real(8):: s

      do i = 1, 10
         a(i) = i
      enddo

      s = sum(a, 10)

      write(*,*) 's =', s

      end program array_sum

      real function sum(a, n)
      implicit none
      integer, intent(IN):: n
      real(8), intent(IN):: a(10)

      integer:: i
      real(8):: s

      s = 0.0
      do i = 1, 20
         s = s + a(i)
      enddo
      sum = s
      end function sum
