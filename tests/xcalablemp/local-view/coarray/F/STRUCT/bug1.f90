  program get1

    type xxx
       character(len=3) cc(2,2)    !! 12  boundary 3 (may not the best)
    end type xxx

    type zzz
       integer n          !!  4   boundary 4
       type(xxx) x        !! 12   boundary 8 (for inner structures)
       real*8 a(3)        !! 24   boundary 8
       character name     !!  1   boundary 1
    end type zzz

    type(zzz), save :: b(2)[*]

    write(*,*) "sizeof(b)=",sizeof(b)

  end program
