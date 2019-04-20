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

subroutine chk_int2(ierr)
integer irank, xmp_node_num, ierr

!$xmp reduction (max:ierr)
  irank=xmp_node_num()
  if ( irank .eq. 1)then
    if ( ierr .eq. 0 ) then
       write(*,*) "PASS"
    else
       write(*,*) "ERROR"
       call abort
    endif
  endif
return
end subroutine

subroutine chk_int3(ierr, n)
integer ierr
!$xmp nodes p(*)

!$xmp task on p(n)
  if ( ierr .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call abort
  endif
!$xmp end task

return
end subroutine
