  include "xmp_coarray.h"
  !$xmp nodes nd(2)
  !$xmp template t(29)
  !$xmp distribute t(block) onto nd
!
  real(4) :: a(29,4)
  !$xmp align a(i,*) with t(i)
  real(4) :: b(15,4)[*]
  real(4) :: val(15,4)

  b=0.0
  sync all

  !$xmp loop on t(i)
  do i=1,29
     locali = mod(i-1,15)+1
     a(i,1:4)=1.0
!!     b(locali,3)[3-this_image()]=this_image()*1.0
     b(locali,1)=this_image()*1.0
  enddo
  sync all

!--------------------- check
  me=this_image()
  nerr=0
  val=0.0

  if (me==1) then
     ilb=1
     iub=15
  else if (me==2) then
     ilb=1
     iub=14
  endif

  do i=ilb,iub
     if (me==2) then
        val(i,3)=1*1.0
     else if (me==1) then
        val(i,3)=2*1.0
     endif
     val(i,1)=me*1.0
  enddo

  do i=1,15
     if (b(i,j).ne.val(i,j)) then
        nerr=nerr+1
        write(*,101) i,j,me,b(i,j),val(i,j)
     endif
  enddo

  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

101 format ("b(",i0,",",i0,")[",i0,"]=",f8.3," should be ",f8.3)

  end program
