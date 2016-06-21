  integer atm[*]

  atm=0
  syncall

  if (this_image()==1) then
     call atomic_define(atm[2], 111)
     atm[3]=222
  endif

  syncall

  write(*,*) this_image(), atm
  end
