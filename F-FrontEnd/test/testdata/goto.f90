      program main
        integer :: i = 0
10      i = i + 10
        if (i >= 10) goto 20
        goto 10
20      i = i + 10
        goto (10, 20, 30) i
30      i = i + 20
      end
