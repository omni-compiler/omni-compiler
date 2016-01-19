program bug468
  include "xmp_coarray.h"
  real,allocatable :: hoge(:)[:]
  real :: tmp, pon(10)[*], eps=0.00001
  tmp=100.0

  !! TESTS --------------------------------
  allocate(hoge(1)[*])

  hoge(1) = tmp     !! scalar assignemnt to coarray

  pon = hoge(1)     !! bcast assignemnt to coarray


  !! CHECK --------------------------------
  nerr=0
  do i=1,10
     if (pon(i)-real(100+i)>eps) then
        nerr=nerr+1
     endif
  enddo

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)',  this_image(), nerr
     stop 1
  end if

end program 
