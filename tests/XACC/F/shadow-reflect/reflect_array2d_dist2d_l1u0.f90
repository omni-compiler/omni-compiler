program aa
  integer, parameter :: ly = 6
  integer, parameter :: lz = 6

  !$XMP NODES p(2,1)
  !$XMP TEMPLATE t(ly,lz)
  !$XMP DISTRIBUTE t(BLOCK,BLOCK) ONTO p
  real*8 :: array(ly,lz)
  !$XMP ALIGN (j,k) WITH t(j,k) :: array
  !$XMP SHADOW (1:0,1:0) :: array
  integer :: err = 0

  !$xmp task on p(1,1)
  do iy = 0, 3 !0|1,2,3
     do iz = 1, 6
        array(iy,iz) = 1.0
     end do
  end do

  !$xmp end task
  !$xmp task on p(2,1)
  do iy = 3, 6 !3|4,5,6
     do iz = 1, 6
        array(iy,iz) = 2.0
     end do
  end do
  !$xmp end task

  !$acc data copy(array)
  !$XMP REFLECT (array) width(1:0,0) acc
  !$acc end data

  !$xmp task on p(2,1)
  do iy = 3, 6
     do iz = 1, 6
        if(iy >= 4 .and. iy <= 6) then
           if(array(iy,iz) /= 2.0) then
              err = err + 1
           end if
        else
           if(array(iy,iz) /= 1.0) then
              err = err + 1
           end if
        end if
!        write(*,*) iy, iz, array(iy,iz)
     end do
  end do
  !$xmp end task

  !$xmp reduction(+:err)

  if(err > 0) then
     call exit(1)
  end if

  !$xmp task on p(1,1)
  write(*,*) "PASS"
  !$xmp end task

end program aa
