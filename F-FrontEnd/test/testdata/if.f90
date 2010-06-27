      program main
        integer :: i = 1
        if (i > 0) then
          i = 2
        else if (i > 10) then
          i = 3
        else
          i = 4
        endif
      end
