subroutine ixmp_sub()
  integer irank, xmp_node_num
!$xmp nodes p(4)

!$xmp task on p(2)
  irank=xmp_node_num()
  if(irank == 1)then
    print *, "PASS"
  else
    print *, "ERROR rank=",irank
    call exit(1)
  end if
!$xmp end task

end subroutine
