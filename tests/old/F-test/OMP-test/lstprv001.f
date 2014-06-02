      program main
          integer a, i, nt
c$        external omp_get_thread_num
c$        integer omp_get_thread_num
c$        external omp_get_max_threads
c$        integer omp_get_max_threads

          a = 0
          nt = 1
c$        nt = omp_get_max_threads()

c$omp parallel do default(shared) lastprivate(a) private(i)
          do i = 1, nt
              a = 1
          end do
c$omp end parallel do

          if (a .eq. 1) then
              print *, "lstprv001 PASSED"
          else
              print *, "lstprv001 FAILED"
          end if

      end

