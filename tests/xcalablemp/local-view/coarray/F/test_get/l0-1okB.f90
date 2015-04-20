  include "xmp_coarray.h"

  logical*2 l1[*]
  logical l2

  me=this_image()

  l1=.true.
  l2=.false.
  syncall

  if (me==3) then
     l2=l1[2]
  endif
  syncall

  nerr=0
  if (me==3) then
     if (.not.l2) then         ! l2 must be .true.
        nerr=nerr+1
        write(*,*) "[",me,"] l2 must be .true. but l2=",l2
     endif
  else
     if (l2) then           ! l2 must be .false.
        nerr=nerr+1
        write(*,*) "[",me,"] l2 must be .false. but l2=",l2
     endif
  endif

  if (nerr==0) then
     write(*,100) me
  else
     write(*,101) me,nerr
  endif

100 format("[",i0,"] OK")
101 format("[",i0,"] NG nerr=",i0)
  end
