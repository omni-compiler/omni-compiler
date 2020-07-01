  integer*2 a1d1(10)[*], tmp(10)

  if (xmpf_coarray_uses_fjrdma()) then
     write(*,'(a)') "Using FJRDMA ... stop"
     stop
  endif

!--------- init
  me = xmp_node_num()
  do i=1,10
     a1d1(i) = i*me+max(-1,0)
     tmp(i) = i+me
  enddo
  sync all

!--------- exec
!!  if (me==2) a1d1(2:a1d1(3)[2_2])[3] = tmp(2:6)
  if (me==2) a1d1(2:a1d1(3)[2])[3] = tmp(2:6)
  sync all

!--------- check
  nerr=0
  do i=1,10
     if (me.eq.3.and.i.ge.2.and.i.le.6) then
        ival = i+2
     else
        ival = i*me
     endif
     if (a1d1(i).ne.ival) then
        write(*,101) i,me,a1d1(i),ival
        nerr=nerr+1
     end if
  enddo

101 format ("a1d1(",i0,")[",i0,"]=",i0," should be ",i0)

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

end program

