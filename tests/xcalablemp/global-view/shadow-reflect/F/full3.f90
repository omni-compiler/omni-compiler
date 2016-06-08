program test
  call sub
end program test

subroutine sub

!$xmp nodes p(2,2)

!$xmp template t(10,10)
!$xmp distribute t(block,block) onto p

  integer b(10,10)
!$xmp align b(i,j) with t(i,j)
!$xmp shadow b(0,*)

!$xmp reflect (b)

!$xmp task on p(1,1)
  write(*,*) 'PASS'
!$xmp end task
  
end subroutine sub
