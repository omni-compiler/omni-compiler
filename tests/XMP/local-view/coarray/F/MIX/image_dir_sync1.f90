PROGRAM zzz

  !$xmp nodes p(8)

  integer aa(4)[*]
  integer aaok(4)
  !$xmp coarray on p :: aa

  me=this_image()

  do i=1,4
     aa(i)=me*10+i
  enddo
  sync all

  !$xmp tasks
    !$xmp task on p(1:4)
       me1 = this_image()
       ntmp = aa(1)[me1+4]
       !$xmp image(p)
          sync images(me1+4)
       aa(2)[me1+4] = ntmp
    !$xmp end task

    !$xmp task on p(5:8)
       me2 = this_image()
       ntmp = aa(1)[me2]
       !$xmp image(p)
          sync images(me2)
       aa(2)[me2] = ntmp
    !$xmp end task
  !$xmp end tasks

  sync all

!-------------------------- check
  nerr=0

  do i=1,4
     aaok(i)=me*10+i
  enddo

  select case(me)
  case(5)
     aaok(2)=51
     ntmpok=11
  case(6)
     aaok(2)=61
     ntmpok=21
  case(7)
     aaok(2)=71
     ntmpok=31
  case(8)
     aaok(2)=81
     ntmpok=41
  case(1)
     aaok(2)=11
     ntmpok=51
  case(2)
     aaok(2)=21
     ntmpok=61
  case(3)
     aaok(2)=31
     ntmpok=71
  case(4)
     aaok(2)=41
     ntmpok=61
  end select

  do i=1,4
     if (aa(i) /= aaok(i)) then
        nerr=nerr+1
        write(*,210) me,i,aaok,aa
     endif
  enddo
  if (ntmp /= ntmpok) then
     nerr=nerr+1
     write(*,200) me,ntmpok,ntmp
  endif

200 format("[",i1,"] ntmp should be ",i2," but ",i2)
210 format("[",i1,"] aa(",i1,") should be ",i2," but ",i2)

  if (nerr==0) then
     write(*,100) me
  else
     write(*,110) me, nerr
  end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

END PROGRAM zzz
