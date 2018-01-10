program main
  include 'xmp_lib.h'
!$xmp nodes p(*)
!$xmp template t0(10)
!$xmp template t1(10)
!$xmp distribute t0(block) onto p
!$xmp distribute t1(block) onto p
  real*4 a(10)
!$xmp align a(i) with t0(i)

!$xmp loop (i) on t1(i)
  do i=1, 10
     a(i) = i
  end do

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task
end program main
