program thisimage
!! !!  include "xmp_coarray.h"

  integer, parameter :: ndx=2, ndy=3, ndz=2
  integer me1,mez1,mey1,mex1,tmp,nerr
  integer iop(3), im_a(3),im_a1,im_a2,im_a3
  real a[ndx,ndy,*]

  me1 = this_image() - 1
  mez1 = me1/(ndx*ndy)
  tmp = me1 - mez1*ndx*ndy
  mey1 = tmp/ndx
  mex1 = tmp - mey1*ndx
  iop(1) = mex1
  iop(2) = mey1
  iop(3) = mez1

  im_a = this_image(a)
  im_a1 = this_image(a,1)
  im_a2 = this_image(coarray=a,dim=2)
!!!bug309  im_a3 = this_image(dim=3+0,CoArray=A)
  im_a3 = this_image(dim=3,CoArray=A)

  nerr = 0

  if (im_a(1).ne.iop(1)+1.or.im_a(2).ne.iop(2)+1.or.im_a(2).ne.iop(2)+1) then
     nerr=nerr+1
     write(*,100) "this_image(a)", iop+1, im_a
  endif 
  if (im_a1.ne.iop(1)+1) then
     nerr=nerr+1
     write(*,101) "this_image(a,1)", iop(1)+1, im_a1
  endif 
  if (im_a2.ne.iop(2)+1) then
     nerr=nerr+1
     write(*,101) "this_image(coaray=a,dim=2)", iop(2)+1, im_a2
  endif 
  if (im_a3.ne.iop(3)+1) then
     nerr=nerr+1
!!     write(*,101) "this_image(DIM=1+2,CoArray=A)", iop(3)+1, im_a3
     write(*,101) "this_image(DIM=3,CoArray=A)", iop(3)+1, im_a3
  endif 

100 format(a," must be (",i0,",",i0,",",i0,") but (",i0,",",i0,",",i0,")")
101 format(a," must be ",i0," but ",i0)

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if

end program thisimage
