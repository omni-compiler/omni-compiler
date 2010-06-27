      program main
          implicit none
          integer i, j, c, n, err
          parameter(n = 100)
          err = 0
          c = 0

c
c i, j must be private
c

c$omp parallel firstprivate(c) shared(err)
          do i = 1, n
              do j = 1, n
                  c = c + 1
                  if (i .eq. n / 2) then
c.....................print for yield
                      print *, "half"
                  end if
                  if (j > n) then
                      print *, "err#1: j=", j
                      err = 1
                      exit
                  end if
              end do
              if (i > n) then
                  print *, "err#2: i=", i
                  err = 1
                  exit
              end if
          end do

          if (c .ne. n * n) then
              print *, "err#3: c=", c
              err = 1
          end if

c$omp end parallel

          if (err .eq. 0) then
              print *, "dflt001 PASSED"
          else
              print *, "dflt001 FALIED"
          end if

      end program

