      program main
        integer, pointer, dimension(:,:) :: left
        integer, target, dimension(4,3) :: right

        right = 4

        left => right
      end program main
