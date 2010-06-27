      program main
        integer sum, i
        sum = 0
        label1: do i = 1,10
          if (i == 3) cycle label1
          sum = sum + i
          if (sum > 100) exit label1
          continue
        end do label1
      end
