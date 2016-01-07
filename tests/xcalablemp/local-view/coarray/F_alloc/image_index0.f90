  include "xmp_coarray.h"
  real, allocatable :: a(:,:)[:,:,:]
  allocate (a(1:2,3:5)[6:7,8:10,-3:*])      !! 2x3xn nodes expected
  nerr=0
  n8 = image_index(a, [7,8,-2])   !! 2 + 2*0 + 2*3*1 = 8  if ni>=8  else 0
  n3 = image_index(a, [6,9,-3])   !! 1 + 2*1 + 2*3*0 = 3  if ni>=3  else 0
  n21 = image_index(a, [6,9,0])   !! 1 + 2*1 + 2*3*3 = 21 if ni>=21 else 0

  n8OK = 8
  n3OK = 3
  n21OK = 21
  ni = num_images()
  if (ni<8) n8OK=0
  if (ni<3) n3OK=0
  if (ni<21) n21OK=0

  nerr=0
  if (n8.ne.n8OK) nerr=nerr+1
  if (n3.ne.n3OK) nerr=nerr+1
  if (n21.ne.n21OK) nerr=nerr+1

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
