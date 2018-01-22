  real :: a(1:2,3:5)[6:*]
  nnnn = image_index(a, [8])
  if (nnnn==3) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] NG: nnnn=",i0)', this_image(), nnnn
  end if
  end
