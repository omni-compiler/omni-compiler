program loop
integer i
integer,parameter :: n=8
integer a(n)
!$xmp nodes p(2)
!$xmp template tx(n+3)
!$xmp distribute tx(cyclic(2)) onto p
!$xmp align a(i) with tx(i+3)

!$xmp loop (i) on tx(i+3)
do i=1,n
  a(i)=i
end do

!$xmp task on p(1)
write(*,*) "PASS"
!$xmp end task

end program loop
