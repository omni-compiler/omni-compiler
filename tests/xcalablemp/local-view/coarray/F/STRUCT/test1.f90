  program get1

    type zzz
       integer n
       real a
       real, allocatable :: b(:)
    end type zzz

    type(zzz) :: x,y

    write(*,*) sizeof(y%b), sizeof(y)

    y%n=111
    y%a=0.001
    allocate (y%b(10))
    y%b = 99.999
    write(*,*) sizeof(y%b)

    x%n=333
    x%a=1.234
    allocate (x%b(3))
    x%b=(/1.0,2.0,3.0/)

    y=x
    write(*,*) y%n, y%a, y%b

    write(*,*) sizeof(y%b), sizeof(y)

  end program
