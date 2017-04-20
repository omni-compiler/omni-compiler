#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

  !$xmp nodes p(8)
  !$xmp nodes q(5)=p(2:6)

  integer me, n1, n2, n3[*]

  me = this_image()
  n1 = me+10
  n2 = me+20
  n3 = me+30

  sync all

block
  !$xmp nodes q(5)=p(3:7)

  integer, save :: me, n1, n2, n3[*]

  me = this_image()
  n1 = me+40
  n2 = me+50
  n3 = me+60

  sync all

  if (3<=me.and.me<=7) then
     !$xmp image(q)
     sync all

     if (me==3) then
        sync images(4)
     else if (me==4) then
        !$xmp image(q)
        sync images(1)      !! initial image 3
     endif

  endif

  sync all

  nerr=0

  if (n2.ne.me+50) then
     nerr = nerr+1
     write(*,200) me, "n2", me+20, n2
  endif
  if (n3.ne.me+60) then
     nerr = nerr+1
     write(*,200) me, "n3", me+30, n3
  endif

  if (nerr==0) then
     write(*,100) me
  else
     write(*,110) me, nerr
  end if

end block

  if (2<=me.and.me<=6) then
     !$xmp image(q)
     sync all

     if (me==2) then
        sync images(3)
     else if (me==3) then
        !$xmp image(q)
        sync images(1)      !! initial image 2
     endif

  endif

  sync all

  nerr=0

  if (n2.ne.me+20) then
     nerr = nerr+1
     write(*,200) me, "n2", me+20, n2
  endif
  if (n3.ne.me+30) then
     nerr = nerr+1
     write(*,200) me, "n3", me+30, n3
  endif

  if (nerr==0) then
     write(*,100) me
  else
     write(*,110) me, nerr
  end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)
200 format("[",i1,"] ",a," should be ",i0," but ",i0)

#else
print *, 'SKIPPED'
#endif
end program

