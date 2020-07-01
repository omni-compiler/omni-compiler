program test

!$xmp nodes p(4,4)

!$xmp template t(64,64)
!$xmp distribute t(block,block) onto p

  integer :: a(64,64)
!$xmp align a(i,j) with t(i,j)
!$xmp shadow a(1,1)

  integer :: x(0:,0:)[*]
!$xmp local_alias x => a
  
  integer :: b(64,64)
!$xmp align b(i,j) with t(i,j)
!$xmp shadow b(1,1)

  integer :: result = 0

  integer :: ierr
  
  integer :: p1, p2
  integer :: n, s, w, e
  integer :: nn, ss, ww, ee
  
  !$xmp loop on t(i,j)
  do j = 1, 64
     do i = 1, 64
        a(i,j) = i * 100 + j
        b(i,j) = i * 100 + j
     end do
  end do

  ierr = xmp_nodes_index(xmp_desc_of(p), 1, p1)
  ierr = xmp_nodes_index(xmp_desc_of(p), 2, p2)

  n = mod(p1-1-1 + 4, 4) + 1
  s = mod(p1-1+1 + 4, 4) + 1
  w = mod(p2-1-1 + 4, 4) + 1
  e = mod(p2-1+1 + 4, 4) + 1

  nn = (n -1) + (p2-1) * 4 + 1
  ss = (s -1) + (p2-1) * 4 + 1
  ww = (p1-1) + (w -1) * 4 + 1
  ee = (p1-1) + (e -1) * 4 + 1
  
  sync all
  
  do j = 1, 16
     x(17,j)[nn] = x(1,j)
     x(0,j)[ss]  = x(16,j)
  end do
  
  do i = 1, 16
     x(i,17)[ww] = x(i,1)
     x(i,0)[ee]  = x(i,16)
  end do

  sync all
  
!$xmp reflect (b) width(/periodic/1:1, /periodic/1:1)

!$xmp loop on t(i,j) reduction(+:result)
  do j = 1, 64
     do i = 1, 64

        if (a(i-1,j) /= b(i-1,j)) then
           result = 1
        end if

        if (a(i,j-1) /= b(i,j-1)) then
           result = 1
        end if

        if (a(i,j) /= b(i,j)) then
           result = 1
        end if

        if (a(i+1,j) /= b(i+1,j)) then
           result = 1
        end if

        if (a(i,j+1) /= b(i,j+1)) then
           result = 1
        end if

     end do
  end do

!$xmp task on p(1,1)
  if (result == 0) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program test

! program test

!   !$xmp nodes p(4,4)

!   !$xmp template t(16,16)
!   !$xmp distribute t(block,block) onto p

!   integer a(16,16)
!   !$xmp align a(i,j) with t(i,j)
!   !$xmp shadow a(1,1)
  
!   integer :: b(0:,0:)[*]
!   !$xmp local_alias b => a

!   !$xmp loop on t(i,j)
!   do j = 1, 16
!      do i = 1, 16
!         a(i,j) = i
!   end do

!   me = this_image()
!   n = num_images()

!   sync all
  
!   b(2)[mod(me,n)+1] = me * 100
!   !b(2) = me * 100

!   sync all
  
!   !$xmp loop on t(i)
!   do i = 1, 16
!      write(*,*) i, a(i)
!   end do

! end program test
  
