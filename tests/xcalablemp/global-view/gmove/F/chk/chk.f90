subroutine chk_int(ierr)
integer ierr
!$xmp nodes p(*)

!$xmp task on p(1)
  if ( ierr .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call abort
  endif
!$xmp end task

return
end subroutine
