program main
  include 'xmp_lib.h'
  integer, parameter:: N=10
  integer, allocatable :: a(:)
  integer error
!$xmp nodes p(3)

  allocate(a(10))
  
  error = 0

!! Check PLUS
  do i=1, 10
     a(i) = xmp_node_num()*100 + i
  end do
  
!$xmp reduction(+:a)
  do i=1, 10
     if(a(i) .ne. (100+i)+(200+i)+(300+i) ) then
        error = 1
     end if
  end do

!! Check Mult
  do i=1, 10
     a(i) = xmp_node_num()*100 + i
  end do
!$xmp reduction(*:a)
  do i=1, 10
     if(a(i) .ne. (100+i)*(200+i)*(300+i) ) then
        error = 1
     end if
  end do

!! Check Max
  do i=1, 10
     a(i) = xmp_node_num()*100 + i
  end do
!$xmp reduction(max:a)
  do i=1, 10
     if(a(i) .ne. (300+i) ) then
        error = 1
      end if
   end do

!! Check Min
  do i=1, 10
     a(i) = xmp_node_num()*100 + i
  end do
!$xmp reduction(min:a)
  do i=1, 10
     if(a(i) .ne. (100+i) ) then
        error = 1
     end if
  end do

!$xmp reduction(+:error)
!$xmp task on p(1)
  if( error .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task
end program main
