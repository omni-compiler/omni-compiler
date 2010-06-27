      program main
        integer sum, i
        sum = 0
        do i = 1,10
          if (i == 3) cycle
          sum = sum + i
          if (sum > 100) exit
        end do
      end
