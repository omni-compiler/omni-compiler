  include "xmp_coarray.h"
  real a(1:2,3:5)[6:7,8:10,-3:*]
  n1 = this_image(a, 1)
  n2 = this_image(a, 2)
  n3 = this_image(a, 3)

  nerr=0
  me=this_image()
  if (me==1) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.8) nerr=nerr+1
     if (n3.ne.-3) nerr=nerr+1
  else if (me==2) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.8) nerr=nerr+1
     if (n3.ne.-3) nerr=nerr+1
  else if (me==3) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.9) nerr=nerr+1
     if (n3.ne.-3) nerr=nerr+1
  else if (me==4) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.9) nerr=nerr+1
     if (n3.ne.-3) nerr=nerr+1
  else if (me==5) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.10) nerr=nerr+1
     if (n3.ne.-3) nerr=nerr+1
  else if (me==6) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.10) nerr=nerr+1
     if (n3.ne.-3) nerr=nerr+1
  else if (me==7) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.8) nerr=nerr+1
     if (n3.ne.-2) nerr=nerr+1
  else if (me==8) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.8) nerr=nerr+1
     if (n3.ne.-2) nerr=nerr+1
  else if (me==9) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.9) nerr=nerr+1
     if (n3.ne.-2) nerr=nerr+1
  else if (me==10) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.9) nerr=nerr+1
     if (n3.ne.-2) nerr=nerr+1
  else if (me==11) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.10) nerr=nerr+1
     if (n3.ne.-2) nerr=nerr+1
  else if (me==12) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.10) nerr=nerr+1
     if (n3.ne.-2) nerr=nerr+1
  else if (me==13) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.8) nerr=nerr+1
     if (n3.ne.-1) nerr=nerr+1
  else if (me==14) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.8) nerr=nerr+1
     if (n3.ne.-1) nerr=nerr+1
  else if (me==15) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.9) nerr=nerr+1
     if (n3.ne.-1) nerr=nerr+1
  else if (me==16) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.9) nerr=nerr+1
     if (n3.ne.-1) nerr=nerr+1
  else if (me==17) then
     if (n1.ne.6) nerr=nerr+1
     if (n2.ne.10) nerr=nerr+1
     if (n3.ne.-1) nerr=nerr+1
  else if (me==18) then
     if (n1.ne.7) nerr=nerr+1
     if (n2.ne.10) nerr=nerr+1
     if (n3.ne.-1) nerr=nerr+1
  end if

  call final_msg(nerr)

  end


  subroutine final_msg(nerr)
    include 'xmp_coarray.h'
    if (nerr==0) then 
       print '("[",i0,"] OK")', this_image()
    else
       print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
    end if
    return
  end subroutine final_msg
