program xacc_distarray
  implicit none
  integer,parameter :: XSIZE = 1024
  integer,parameter :: YSIZE = 512
  integer u(XSIZE), v(XSIZE, YSIZE)
  integer x, y
  logical flag = .true.
  integer sum = 0

  !$xmp nodes p(4)
  !$xmp template t(XSIZE)
  !$xmp distribute t(block) onto p
  !$xmp align u(x) with t(x)
  !$xmp align v(*, y) with t(y)

  !init
  !$xmp loop (x) on t(x)
  do x = 1, XSIZE
     u(x) = 0
  end do
  
  !$xmp loop (y) on t(y)
  do y = 1, YSIZE
     do x = 1, XSIZE
        v(x, y) = 0
     end do
  end do

  !test
  !$acc data copy(u, v(:,:))

  !$xmp loop (x) on t(x)
  !$acc parallel loop
  do x = 1, XSIZE
     u(x) = x
  end do

  !$xmp loop (y) on t(y)
  !$acc parallel loop
  do y = 1, YSIZE
     do x = 1, XSIZE
        v(x, y) = x + y
     end do
  end do

  !$xmp loop (y) on t(y)
  !$acc parallel loop reduction(+:sum)
  do y = 1, YSIZE
     sum = sum + y
  end do

  !$acc end data

  !$xmp reduction(+:sum)

  !check result
  !$xmp loop (x) on t(x)
  do x = 1, XSIZE
     if (u(x) .ne. x) then
        flag = .false.
     end if
  end do

  !$xmp loop (y) on t(y)
  do y = 1, YSIZE
     do x = 1, XSIZE
        if (v(x, y) .ne. x + y) then
           flag = .false.
        end if
     end do
  end do

  if(sum .ne. YSIZE * (YSIZE + 1) / 2) then
     flag = .false.
  end if

  !$xmp task on p(1)
  if (flag) then
     print *,"OK"
  else
     print *,"invalid result"
  end if
  !$xmp end task
end program distarray
