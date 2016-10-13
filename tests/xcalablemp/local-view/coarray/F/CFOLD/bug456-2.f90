parameter(lx=100,ly=lx/2)
! parameter(lx=100,ly=50)
!! include 'xmp_coarray.h'
real*8 :: a(lx/10,-ly*(-2))[ly,*]

if (this_image() == 1) then
   do i = 1, lx/10           !! 10
      do j = 1, -ly*(-2)     !! 100
         a(i,j) = i+j*10
      end do
   enddo
endif

sync all

if (this_image() == 2) then
   a[3,1] = a[1,1]
endif 

sync all

nerr = 0
if (this_image() == 3) then
   do i = 1, lx/10           !! 10
      do j = 1, -ly*(-2)     !! 100
         if (a(i,j) /= i+j*10) then
            nerr = nerr+1
         endif
      end do
   enddo
endif

  if (nerr==0) then 
     print '("[",i0,"] OK")', this_image()
  else
     print '("[",i0,"] number of NGs: ",i0)', this_image(), nerr
  end if


stop
end
