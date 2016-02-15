program bug462
  include 'xmp_coarray.h'
  parameter (n1=17,n2=1000,n3=1000)
  integer a(n1,n2,n3)[*]   !! 4*17*1000*1000 > 16M
  integer b(n1,n2,n3)

  !----------------------- init
  me=this_image()
  a=152
  b=375
  syncall

  !----------------------- exec
  if (me==2) a(:,:,:)[3]=b(:,:,:)
  syncall

  !----------------------- check
  if (me==3) then
     nok=375
  else
     nok=152
  endif

  nerr=0
  do 100 k=1,n3
     do 100 j=1,n2
        do 100 i=1,n1
100        if (a(i,j,k).ne.nok) nerr=nerr+1

  !----------------------- output
  if (nerr==0) then 
     print '("[",i0,"] OK")', me
  else
     print '("[",i0,"] number of NGs: ",i0)', me, nerr
  end if

  end
