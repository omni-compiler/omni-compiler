  program struct_pointer

    type g1
       integer a(3)
       type(g1),pointer :: p
    end type g1

    type(g1), save ::  cs1[*]

    me = this_image()

    cs1%a = 0
    if (me==1) cs1%a(1) = 333
    syncall
    if (me==2) cs1%a(2) = cs1[1]%a(1) + 1
    if (me==2) cs1[3]%a(3) = cs1%a(2) + 1
    syncall

    !!---- check
    nerr = 0
    do i=1,3
       nok = 0
       if (me==1.and.i==1) nok = 333
       if (me==2.and.i==2) nok = 334
       if (me==3.and.i==3) nok = 335
       if (cs1%a(i).ne.nok) then
          nerr=nerr+1
          write(*,200) me, i, 333, cs1%a(i)
       endif
    enddo

200 format("[", i0, "] cs1%a(",i0,") should be ",i0," but ",i0)

  if (nerr==0) then
     write(*,100) me
  else
     write(*,110) me, nerr
  end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

  end program
