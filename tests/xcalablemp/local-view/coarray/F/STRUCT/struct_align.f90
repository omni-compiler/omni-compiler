  program struct_align

    type xxx                       !!  boundary  length   proceed
       character(len=3) cc(2,3)    !!           3*2*3=18    18
       integer nn(2)               !!     4      4*2=8      28
    end type xxx                   !!     4                 28

    type zzz                       !!  boundary  length   proceed
       integer n                   !!                4       4
       real*8 a(3)                 !!     8      8*3=24     32
       type(xxx) x                 !!     4         28      60
       character name              !!     8                 64
    end type zzz

    type(xxx), save :: a[*]            !! length   28
    type(xxx), save :: a2(10,20)[*]    !! length 5600

    type(zzz), save :: b(2)[*]         !! length  128

    call check_and_msg(sizeof(a)+sizeof(a2)+sizeof(b))
  end program


  subroutine check_and_msg(nlen)
    integer*8 nlen

    me = this_image()
    n_alloced = xmpf_coarray_allocated_bytes()
    n_necessary = nlen

    if (n_alloced == n_necessary) then
       write(*,100) me
    else if (n_alloced > n_necessary) then
       write(*,101) me, n_alloced, n_necessary
    else  !! NG
       write(*,102) me, n_alloced, n_necessary
    endif

100 format("[", i0, "] OK. perfect")
101 format("[", i0, "] OK, but allocated size (", i0,       &
         " bytes) is larger than necessary size (", i0,     &
         " bytes).")
102 format("[", i0, "] NG. Allocated size (", i0,           &
         " bytes) is smaller than necessary size (", i0,    &
         " bytes).")

  end function check_and_msg
