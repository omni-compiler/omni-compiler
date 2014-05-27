      program main
        integer n, m
C$OMP parallel do reduction(+:n,m) private(i)
        do i = 1, 4
            n = n + i
            m = m + i
            print *, "i=", i
        end do
C$OMP end parallel do

        print *,n, m
      end

