  program test_bug1
    integer a2(0:9)

    a2(:2)=a2(2:4)

  end program

!! report #353
