  program main
    ! Branch to statement labeled 10
    IF (i) 10, 20, 30
    stop
10  print *, "minus"
    stop
20  print *, "zero"
    stop
30  print *, "plus"
  end program main
