      program main
        integer sum, i
        sum = 0
        i = 1
top:        do while (i <= 10)
           if (i == 3) cycle
           if (sum > 100) exit
        sum = sum + i
        i = i + 1          
        continue 
        end do top
      end
