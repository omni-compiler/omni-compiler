!!   include "xmp_coarray.h"

  logical l1[*]
  logical l2
  logical l3[*]

  me=this_image()

  l1=.true.
  l2=.false.
  l3=.false.
  syncall

  if (me==3) then
     call atomic_ref(l2,l1[2])
     call atomic_ref(l3,l1[2])
  endif
  syncall

  nerr=0
  if (me==3) then
     if (.not.l2) then
        nerr=nerr+1
        write(*,200) me,"l2",.true.,l2
     endif
     if (.not.l3) then
        nerr=nerr+1
        write(*,200) me,"l3",.true.,l3
     endif
  else
     if (l2) then
        nerr=nerr+1
        write(*,200) me,"l2",.false.,l2
     endif
     if (l3) then
        nerr=nerr+1
        write(*,200) me,"l3",.false.,l3
     endif
  endif

200 format("[",i0,"] ",a," must be ",l1," but ",l1)

  if (nerr==0) then
     write(*,100) me
  else
     write(*,101) me,nerr
     call exit(1)
  endif

100 format("[",i0,"] OK")
101 format("[",i0,"] NG nerr=",i0)
  end
