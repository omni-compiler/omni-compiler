  integer b(5,3),a(5,3)[*]

  a=1
  b=0

  if (this_image()==1) then
     b(1:5,1:1)=a(1:5,1:1)[2]
  endif

  write(*,101) this_image(),a
  write(*,102) this_image(),b

101 format("var a[",i0,"]=",(i0,i0,i0,i0,i0))
102 format("var b[",i0,"]=",(i0,i0,i0,i0,i0,i0,i0,i0,i0,i0))

end program
