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
      aa(3)[this_image()+4] = aa(1)
      !! p(1): aa(3)[5] = aa(1) = 11
      !! p(2): aa(3)[6] = aa(1) = 21
      !! p(3): aa(3)[7] = aa(1) = 31
      !! p(4): aa(3)[8] = aa(1) = 41
    !$xmp end task
    !$xmp task on p(5:8)
      aa(4)[this_image()] = aa(2)
      !! p(5): aa(4)[1] = aa(2) = 52
      !! p(6): aa(4)[2] = aa(2) = 62
      !! p(7): aa(4)[3] = aa(2) = 72
      !! p(8): aa(4)[4] = aa(2) = 82
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
     aaok(3)=11
  case(6)
     aaok(3)=21
  case(7)
     aaok(3)=31
  case(8)
     aaok(3)=41
  case(1)
     aaok(4)=52
  case(2)
     aaok(4)=62
  case(3)
     aaok(4)=72
  case(4)
     aaok(4)=82
  end select

  do i=1,4
     if (aa(i) /= aaok(i)) then
         nerr=nerr+1
         write(*,210) me,i,aaok,aa
      endif
   enddo

200 format("[",i1,"] ",a2," should be ",i2," but ",i2)
210 format("[",i1,"] aa(",i1,") should be ",i2," but ",i2)

    if (nerr==0) then
       write(*,100) me
    else
       write(*,110) me, nerr
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

  END PROGRAM zzz
