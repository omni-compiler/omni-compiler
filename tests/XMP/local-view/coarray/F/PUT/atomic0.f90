  integer atm[*]

  me = this_image()
  atm=0
  if (me==1) nok=0
  if (me==2) nok=111
  if (me==3) nok=222
  nerr=0

  syncall

  if (me==1) then
     call atomic_define(atm[2], 111)
     atm[3]=222
  endif

  syncall

  if (atm /= nok) then
     nerr=nerr+1
     write(*,200) me, "atm", nok, atm
  endif

  if (nerr == 0) then
     write(*,100) me
  else
     write(*,110) me, nerr
     call exit(1)
  endif

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)
200 format("[",i0,"] ",a," should be ",i0," but ",i0)
  end
