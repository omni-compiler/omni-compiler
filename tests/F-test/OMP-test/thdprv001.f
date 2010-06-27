      program main
          implicit none
          integer t(3)
          logical err
          common /thdprv/ t
          data t /3*999/
c$omp threadprivate(/thdprv/)

          t(1) = 1
          t(2) = 2
          t(3) = 3
          err = .FALSE.

c$omp parallel shared(err) copyin(t)

          if (t(1) .ne. 1) then
              err = .TRUE.
          end if
          if (t(2) .ne. 2) then
              err = .TRUE.
          end if
          if (t(3) .ne. 3) then
              err = .TRUE.
          end if

c$omp end parallel

          if (err) then
              print *, "thdprv001 FAILED"
          else
              print *, "thdprv001 PASSED"
          end if
 
      end program

