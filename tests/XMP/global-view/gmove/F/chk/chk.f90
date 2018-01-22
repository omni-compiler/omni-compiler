subroutine chk_int(tname, ierr)
character(*) tname
integer ierr
!$xmp nodes p(*)

!$xmp task on p(1)
  if ( ierr .eq. 0 ) then
     write(*,*) "PASS ",tname
  else
     write(*,*) "ERROR ",tname
     call abort
  endif
!$xmp end task

return
end subroutine
