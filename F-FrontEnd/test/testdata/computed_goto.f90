      program main
        integer :: i = 2
10      goto (10, 20), i
        print *, "line 10"
20      goto (10, 20, 30), i + 1
        print *, "line 20"
30      i = 2
        print *, "line 30"
      end
