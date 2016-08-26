  program struct_align

    type xxx
       character(len=3) cc(2,3)    !! 18  boundary 3 (may not the best)
       integer nn(2)               !!  8  boundary 4
    end type xxx

    type zzz
       integer n          !!  4   boundary 4
       type(xxx) x        !! 12   boundary 8 (for inner structures)
       real*8 a(3)        !! 24   boundary 8
       character name     !!  1   boundary 1
    end type zzz

    type(xxx), save :: a[*], a2(10,20)[*]

    type(zzz), save :: b(2)[*]

    me = this_image()
    write(*,100) me, sizeof(a),sizeof(a2),sizeof(b)
100 format("[",i0,"] sizeof(a)=",i5," sizeof(a2)=",i5," sizeof(b)=",i5)

  end program
