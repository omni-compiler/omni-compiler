      program main
c i, s ... shared         
c j    ... private         
          integer i, j, s
          s = 0
c s=1*1+2*2+3*3
          do i = 1, 3
c$omp parallel do
              do j = 1, i
c$omp atomic
                  s = s + i
              end do
c$omp end parallel do
          end do
          if (s .eq. 14) then
              print *, "indvar001 PASSED"
          else
              print *, "indvar001 FAILED"
          end if 
      end program

