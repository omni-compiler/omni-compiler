real function foo(n0,a2,m0)
  include "xmp_coarray.h"
!!  use xmp_lib
  integer m0
  integer n0[*]
  real a2(3,m0)[*]

  k=n0[m0]
  foo = k * a2(2,3)[1]
  a2(2,5)[2] = 3.333
  return
end function foo

program dummyarg
  include "xmp_coarray.h"

  integer n[*]
  real a1(300)[*]

  me = this_image()

  !---------------------------- switch on message
!!    if (me==2) call xmpf_coarray_msg(1)

  !---------------------------- initialization
  do i=1,300
     a1(i) = float(me + 10*i)
  end do
  n = - me
  sync all

  !---------------------------- execution
  ans = 0.0
  if (me==3) then
     ans = foo(n,a1,3)
     ! k = n0[m0] = n[3] = -3
     ! foo = k * a2(2,3)[1] = (-3) * a1(2+(3-1)*3)[1]
     !     = (-3) * float(1 + 10*8) = -243.0
  end if

  sync all

  !---------------------------- check and output
  nerr=0
  if (me==3) then
     if (-243.0001 < ans .and. ans < -242.9999) then
        continue
     else
        nerr=nerr+1
     endif
  else
     if (-0.0001 < ans .and. ans < 0.0001) then
        continue
     else
        nerr=nerr+1
     endif
  endif

  eps = 0.001
  do i = 1, 300
     if (me==2.and.i==14) then
        if (3.33299 < a1(i) .and. a1(i) < 3.33301) then
           continue
        else
           nerr=nerr+1
           print *,"a1(",i,")[",me,"]=",a1(i)," should be",3.333
        endif
     else 
        val = float(me + 10*i)
        if (val-eps < a1(i) .and. a1(i) < val + eps) then
           continue
        else
           nerr=nerr+1
           write(*,*) "me=", me
           write(*,*) "a1(",i,")[",me,"]=",a1(i)," should be",val
        endif
     endif
  enddo

  if (nerr==0) then
     print '("result[",i0,"] OK")', me
  else
     print '("result[",i0,"] number of NGs: ",i0)', me, nerr
  end if

  end program
