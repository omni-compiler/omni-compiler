      program end
        integer n(2)
C$OMP parallel do reduction(ieor:n) private(i)
        do i = 1, 4
            n = n + i
            print *, "i=", i
        end do
C$OMP end parallel do
        print *,n
      end

