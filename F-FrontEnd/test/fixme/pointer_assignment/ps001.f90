      program main
        integer, pointer :: left
        integer, target  :: right

        right = 1

        left => right
      end program main
