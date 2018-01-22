  !$xmp nodes nd(2)
  !$xmp template t(129)
  !$xmp distribute t(block) onto nd
!
  real(4) :: a(129,4)
  !$xmp align a(i,*) with t(i)

  !$xmp loop on t(i)
  do i=1,129
     a(i,1:4)=1.0
  enddo

!$xmp loop on t(i)
  do i=1,1
     if (a(i,1).eq.1.0) then
        write(*,*) "PASS"
     else
        write(*,*) "NG"
     endif
  enddo

end program
