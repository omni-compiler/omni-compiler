      program main
        integer,dimension(4,4) :: x
        real :: y
        real,dimension(4,4) :: z

        logical,dimension(4,4) :: la
        logical :: lb



        y = 0.125
        x = 4

        x = x + y
        x = y + x
        x = x + x

        la = x > y

        la = .true.
        lb = .false.

        la = la .and. lb

        print *, x + y

      end program main
