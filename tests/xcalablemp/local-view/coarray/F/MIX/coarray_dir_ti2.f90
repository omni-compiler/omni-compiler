PROGRAM zzz
  implicit none

  !$xmp nodes p(8)
  !$xmp nodes p3(3,2)=p(2:7)

  integer n3[3,*]
  integer nn31, nn32
  integer nn31ok, nn32ok
  integer i, nerr, me

  !$xmp coarray on p3 :: n3

  me = this_image()
  nn31 = this_image(n3,1)
  nn32 = this_image(n3,2)

  select case(this_image())
  case (1)     !! restriction: this_image(n3) is invalid
     nn31ok=0      
     nn32ok=0
  case (2)
     nn31ok=1
     nn32ok=1
  case (3)
     nn31ok=2
     nn32ok=1
  case (4)
     nn31ok=3
     nn32ok=1
  case (5)
     nn31ok=1
     nn32ok=2
  case (6)
     nn31ok=2
     nn32ok=2
  case (7)
     nn31ok=3
     nn32ok=2
  case (8)     !! restriction: this_image(n3) is invalid
     nn31ok=0
     nn32ok=0
  end select

!--------------------------------------------- error check
  nerr = 0
  if (nn31 /= nn31ok) then
     if (nn31ok==0) then
        write(*,220) me,"nn31"
     else
        nerr = nerr + 1
        write(*,200) me,"nn31",nn31ok,nn31
     endif
  end if
  if (nn32 /= nn32ok) then
     if (nn32ok==0) then
        write(*,220) me,"nn32"
     else
        nerr = nerr + 1
        write(*,200) me,"nn32",nn32ok,nn32
     endif
  end if


200 format("[",i0,"] ",a," should be ",i0," but ",i0)
210 format("[",i0,"] ",a,"(",i0,") should be ",i0," but ",i0)
220 format("[",i0,"] RESTRICTION: ",a," is invalid.")

    if (nerr==0) then
       write(*,100) me
    else
       write(*,110) me, nerr
    end if

100 format("[",i0,"] OK")
110 format("[",i0,"] NG nerr=",i0)

END PROGRAM zzz

