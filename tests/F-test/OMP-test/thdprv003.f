      program main
          integer,save :: a(2, -2:3)
          integer lb
c$omp threadprivate(/a/)
          logical,save :: err
          err = .FALSE.
      
c$omp parallel private(lb)
          lb = lbound(a, 1)
          if (lb .ne. 1) then
              err = .TRUE.
          end if

          lb = lbound(a, 2)
          if (lb .ne. -2) then
              err = .TRUE.
          end if
c$omp end parallel

          if (err) then
              print *, "thdprv003 FAILED"
          else
              print *, "thdprv003 PASSED"
          end if

      end program
