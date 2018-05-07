
subroutine check(iresult,ians,error)
  integer iresult,ians,error
  if(iresult .ne. ians) then
    error=error+1
  end if
end subroutine

