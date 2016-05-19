program coarray_dir3
  !$xmp nodes p(4,2)
  !$xmp nodes q1(2,2)=p(1:2,1:2)

  integer a[*], aok

  me0 = this_image()
  me1 = -1
  a = -1
  sync all

  !$xmp task on q1
    me1 = this_image()
    if (me1==1) then
       a[1]=100
       a[2]=200
       a[3]=300
       a[4]=400
    endif
  !$xmp end task
  sync all

!!---- check

  me1ok = -1
  if (me0==1) me1ok=1
  if (me0==2) me1ok=2
  if (me0==5) me1ok=3
  if (me0==6) me1ok=4

  aok = -1
  if (me0==1) aok=100
  if (me0==2) aok=200
  if (me0==5) aok=300
  if (me0==6) aok=400

  nerr=0
  if (me1 /= me1ok) then
     nerr=nerr+1
     write(*,200) me0,"me1",me1ok,me1
  endif
  if (a /= aok) then
     nerr=nerr+1
     write(*,200) me0,"a",aok,a
  endif

200 format("[",i0,"] ",a," should be ",i0," but ",i0)

  if (nerr==0) then
     write(*,100) me0
  else
     write(*,110) me0, nerr
  end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

end program coarray_dir3

