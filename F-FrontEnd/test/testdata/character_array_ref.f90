      program main
        character(LEN=10),dimension(10) :: string

        character(LEN=5), dimension(10) :: substring

        substring(1:2) = string(1:2)(2:6)

      end program main
