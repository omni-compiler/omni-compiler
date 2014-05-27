      subroutine sub 
          implicit none
          integer t(2, 2, 2)
          logical err
          common /thdprv/ t
          common /err/ err
c$omp threadprivate(/thdprv/)

          if (t(1, 1, 1) .ne. 1) then
              err = .TRUE.
          end if
          if (t(1, 2, 1) .ne. 2) then
              err = .TRUE.
          end if
          if (t(2, 1, 1) .ne. 3) then
              err = .TRUE.
          end if
          if (t(2, 2, 1) .ne. 4) then
              err = .TRUE.
          end if
          if (t(1, 1, 2) .ne. 5) then
              err = .TRUE.
          end if
          if (t(1, 2, 2) .ne. 6) then
              err = .TRUE.
          end if
          if (t(2, 1, 2) .ne. 7) then
              err = .TRUE.
          end if
          if (t(2, 2, 2) .ne. 8) then
              err = .TRUE.
          end if

      end subroutine
      
      program main
          implicit none
          integer t(2, 2, 2)
          logical err
          common /thdprv/ t
          common /err/ err
          data t /8*0/
c$omp threadprivate(/thdprv/)

          t(1, 1, 1) = 1
          t(1, 2, 1) = 2
          t(2, 1, 1) = 3
          t(2, 2, 1) = 4
          t(1, 1, 2) = 5
          t(1, 2, 2) = 6
          t(2, 1, 2) = 7
          t(2, 2, 2) = 8

          err = .FALSE.

c$omp parallel copyin(t)

          call sub()

c$omp end parallel

          if (err) then
              print *, "thdprv002 FAILED"
          else
              print *, "thdprv002 PASSED"
          end if
 
      end program

