  program main
    type s1
       integer x
       integer y
    end type s1

    integer, pointer :: pi
    type(s1), dimension(10, 10), target :: s1b

    pi => s1b(1, 1)%x

  end program main

