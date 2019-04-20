PROGRAM zzz
  implicit none

  !$xmp nodes p(8)
  !$xmp nodes p2(2,4)=p

  integer n1[*], n2[2,*]
  integer nn1, nn2(2)
  integer nn1ok, nn2ok(2)
  integer i, nerr, me

  !$xmp coarray on p2 :: n2

  me = this_image()
  nn1 = this_image(n1,1)
  nn2 = this_image(n2)

  select case(this_image())
  case (1)
     nn1ok=1
     nn2ok(1)=1
     nn2ok(2)=1
  case (2)
     nn1ok=2
     nn2ok(1)=2
     nn2ok(2)=1
  case (3)
     nn1ok=3
     nn2ok(1)=1
     nn2ok(2)=2
  case (4)
     nn1ok=4
     nn2ok(1)=2
     nn2ok(2)=2
  case (5)
     nn1ok=5
     nn2ok(1)=1
     nn2ok(2)=3
  case (6)
     nn1ok=6
     nn2ok(1)=2
     nn2ok(2)=3
  case (7)
     nn1ok=7
     nn2ok(1)=1
     nn2ok(2)=4
  case (8)
     nn1ok=8
     nn2ok(1)=2
     nn2ok(2)=4
  end select

!--------------------------------------------- error check
  nerr = 0
  if (nn1 /= nn1ok) then
     nerr = nerr + 1
     write(*,200) me,"nn1",nn1ok,nn1
  end if
  do i=1,2
     if (nn2(i) /= nn2ok(i)) then
        nerr = nerr + 1
        write(*,210) me,"nn2",i,nn2ok(i),nn2(i)
     end if
  end do

200 format("[",i0,"] ",a," should be ",i0," but ",i0)
210 format("[",i0,"] ",a,"(",i0,") should be ",i0," but ",i0)

    if (nerr==0) then
       write(*,100) me
    else
       write(*,110) me, nerr
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

END PROGRAM zzz

