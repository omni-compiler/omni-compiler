program reflect_array4d_dist4d_width2_2
  integer, parameter :: lt = 12
  integer, parameter :: lx = 10
  integer, parameter :: ly = 8
  integer, parameter :: lz = 6

  !$XMP NODES pt(2,1,1,1)
  !$XMP NODES px(1,2,1,1)
  !$XMP NODES py(1,1,2,1)
  !$XMP NODES pz(1,1,1,2)

  !$XMP TEMPLATE tt(lt,lx,ly,lz)
  !$XMP TEMPLATE tx(lt,lx,ly,lz)
  !$XMP TEMPLATE ty(lt,lx,ly,lz)
  !$XMP TEMPLATE tz(lt,lx,ly,lz)

  !$XMP DISTRIBUTE tt(BLOCK,BLOCK,BLOCK,BLOCK) ONTO pt
  !$XMP DISTRIBUTE tx(BLOCK,BLOCK,BLOCK,BLOCK) ONTO px
  !$XMP DISTRIBUTE ty(BLOCK,BLOCK,BLOCK,BLOCK) ONTO py
  !$XMP DISTRIBUTE tz(BLOCK,BLOCK,BLOCK,BLOCK) ONTO pz

  real*8 :: array_t(lt,lx,ly,lz)
  real*8 :: array_x(lt,lx,ly,lz)
  real*8 :: array_y(lt,lx,ly,lz)
  real*8 :: array_z(lt,lx,ly,lz)

  !$XMP ALIGN (it,ix,iy,iz) WITH tt(it,ix,iy,iz) :: array_t
  !$XMP ALIGN (it,ix,iy,iz) WITH tx(it,ix,iy,iz) :: array_x
  !$XMP ALIGN (it,ix,iy,iz) WITH ty(it,ix,iy,iz) :: array_y
  !$XMP ALIGN (it,ix,iy,iz) WITH tz(it,ix,iy,iz) :: array_z

  !$XMP SHADOW (0:2,0:2,0:2,0:2) :: array_t
  !$XMP SHADOW (0:2,0:2,0:2,0:2) :: array_x
  !$XMP SHADOW (0:2,0:2,0:2,0:2) :: array_y
  !$XMP SHADOW (0:2,0:2,0:2,0:2) :: array_z

  integer :: err = 0

  !$xmp task on pt(1,1,1,1)
  do it = 1, 8 !1,2,3,4,5,6|7,8
     do ix = 1, lx
        do iy = 1, ly
           do iz = 1, lz
              array_t(it,ix,iy,iz) = 1.0
           end do
        end do
     end do
  end do
  !$xmp end task
  !$xmp task on px(1,1,1,1)
  do it = 1, lt
     do ix = 1, 7 !1,2,3,4,5|6,7
        do iy = 1, ly
           do iz = 1, lz
              array_x(it,ix,iy,iz) = 1.0
           end do
        end do
     end do
  end do
  !$xmp end task
  !$xmp task on py(1,1,1,1)
  do it = 1, lt
     do ix = 1, lx
        do iy = 1, 6 !1,2,3,4|5,6
           do iz = 1, lz
              array_y(it,ix,iy,iz) = 1.0
           end do
        end do
     end do
  end do
  !$xmp end task
  !$xmp task on pz(1,1,1,1)
  do it = 1, lt
     do ix = 1, lx
        do iy = 1, ly
           do iz = 1, 5 !1,2,3|4,5
              array_z(it,ix,iy,iz) = 1.0
           end do
        end do
     end do
  end do
  !$xmp end task


  !$xmp task on pt(2,1,1,1)
  do it = 7, 12 !7,8,9,10,11,12|13,14
     do ix = 1, lx
        do iy = 1, ly
           do iz = 1, lz
              array_t(it,ix,iy,iz) = 2.0
           end do
        end do
     end do
  end do
  !$xmp end task
  !$xmp task on px(1,2,1,1)
  do it = 1, lt
     do ix = 6, 10 !6,7,8,9,10|11,12
        do iy = 1, ly
           do iz = 1, lz
              array_x(it,ix,iy,iz) = 2.0
           end do
        end do
     end do
  end do
  !$xmp end task
  !$xmp task on py(1,1,2,1)
  do it = 1, lt
     do ix = 1, lx
        do iy = 5, 8 !5,6,7,8|9,10
           do iz = 1, lz
              array_y(it,ix,iy,iz) = 2.0
           end do
        end do
     end do
  end do
  !$xmp end task
  !$xmp task on pz(1,1,1,2)
  do it = 1, lt
     do ix = 1, lx
        do iy = 1, ly
           do iz = 4, 6 !4,5,6|7,8
              array_z(it,ix,iy,iz) = 2.0
           end do
        end do
     end do
  end do
  !$xmp end task

  !$acc data copy(array_t)
  !$XMP REFLECT (array_t) width(0:2,0,0,0) acc
  !$acc end data
  !$acc data copy(array_x)
  !$XMP REFLECT (array_x) width(0,0:2,0,0) acc
  !$acc end data
  !$acc data copy(array_y)
  !$XMP REFLECT (array_y) width(0,0,0:2,0) acc
  !$acc end data
  !$acc data copy(array_z)
  !$XMP REFLECT (array_z) width(0,0,0,0:2) acc
  !$acc end data

  !$xmp task on pt(1,1,1,1)
  do it = 1, 8
     do ix = 1, lx
        do iy = 1, ly
           do iz = 1, lz
              if(1 <= it .and. it <= 6) then
                 if(array_t(it,ix,iy,iz) /= 1.0) then
                    err = err + 1
                 endif
              else
                 if(array_t(it,ix,iy,iz) /= 2.0) then
                    err = err + 1
                 endif
              end if
           end do
        end do
     end do
  end do
  !$xmp end task

  !$xmp reduction(+:err)
  if(err /= 0) then
     call exit(1)
  end if

  !$xmp task on px(1,1,1,1)
  do it = 1, lt
     do ix = 1, 7
        do iy = 1, ly
           do iz = 1, lz
              if(1 <= ix .and. ix <= 5) then
                 if(array_x(it,ix,iy,iz) /= 1.0) then
                    err = err + 1
                 endif
              else
                 if(array_x(it,ix,iy,iz) /= 2.0) then
                    err = err + 1
                 endif
              end if
           end do
        end do
     end do
  end do
  !$xmp end task

  !$xmp reduction(+:err)
  if(err /= 0) then
     call exit(1)
  end if

  !$xmp task on py(1,1,1,1)
  do it = 1, lt
     do ix = 1, lx
        do iy = 1, 6
           do iz = 1, lz
              if(1 <= iy .and. iy <= 4) then
                 if(array_y(it,ix,iy,iz) /= 1.0) then
                    err = err + 1
                 endif
              else
                 if(array_y(it,ix,iy,iz) /= 2.0) then
                    err = err + 1
                 endif
              end if
           end do
        end do
     end do
  end do
  !$xmp end task

  !$xmp reduction(+:err)
  if(err /= 0) then
     call exit(1)
  end if

  !$xmp task on pz(1,1,1,1)
  do it = 1, lt
     do ix = 1, lx
        do iy = 1, ly
           do iz = 1, 5
              if(1 <= iz .and. iz <= 3) then
                 if(array_z(it,ix,iy,iz) /= 1.0) then
                    err = err + 1
                 endif
              else
                 if(array_z(it,ix,iy,iz) /= 2.0) then
                    err = err + 1
                 endif
              end if
           end do
        end do
     end do
  end do
  !$xmp end task

  !$xmp reduction(+:err)
  if(err /= 0) then
     call exit(1)
  end if

  !$xmp task on px(1,1,1,1)
  write(*,*) "PASS"
  !$xmp end task

end program reflect_array4d_dist4d_width2_2
