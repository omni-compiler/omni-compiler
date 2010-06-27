      module test_m
        integer n
      end module

      subroutine sub
        use test_m
        real, dimension(n) :: a

!$omp parallel do private(i,a)
        do i=1,10
          a(i) = 0
        end do
      end

