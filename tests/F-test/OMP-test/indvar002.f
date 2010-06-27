      program main
c s     ... shared         
c i,j,p ... private         

          integer i, j, p, s, err, n
          integer,external::omp_get_thread_num

          err = 0
          s = 0
          n = 100
c s = (1+2+...+n)*n
c$omp parallel do private(p) reduction(+:s)
          do i = 1, n
              p = 0
              do j = 1, n
                  p = p + i
                  if (j > n .and. err .eq. 0) then
                      err = 1
                      print *, "indvar002 FAILED"
                  end if
              end do
              s = s + p
          end do
c$omp end parallel do
          print *, err, s
          if (err .eq. 0 .and. s .eq. n * n * (n + 1) / 2) then
              print *, "indvar002 PASSED"
          else
              print *, "indvar002 FAILED"
          end if 
      end program

