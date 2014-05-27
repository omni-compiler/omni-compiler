      program main
          integer i, a, tn
          integer omp_get_thread_num
          external omp_get_thread_num
          a = 0
c$omp parallel default(private)
          tn = 0
c$        tn = omp_get_thread_num()
c$omp do reduction(+:a)
          do i = 10, 1, -1
              a = a + i 
c             print *,"tn=",tn,",i=",i
          end do
c$omp end do
c$omp end parallel
c         print *,"a=",a
          if(a .eq. 55)then
              print *,"rdctn002 PASSED"
          else
              print *,"rdctn002 FAILED"
          end if
      end program

