      program main
          integer omp_get_thread_num
          external omp_get_thread_num
          integer i, tn
          integer m
          integer a(12), e(12)
          data e /0,0,0,1,1,1,2,2,2,3,3,3/

c$omp parallel default(private) num_threads(4) shared(a)
          tn = 0
c$        tn = omp_get_thread_num()
c$omp do
          do i = 1, 12
              a(i) = tn
          end do
c$omp end do
c$omp end parallel
c         print *,"a=",a
c         print *,"e=",e
          m = 1
          do i = 1, 12
              if (a(i) .ne. e(i)) then
                  m = 0
              end if
          end do

          if(m .eq. 1)then
              print *,"sched001 PASSED"
          else
              print *,"sched001 FAILED"
          end if
      end

