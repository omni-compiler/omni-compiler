subroutine hoge
  !$xmp nodes p(*)
  !$xmp template t(20)
  !$xmp distribute t(block) onto p
  integer :: i, sum, a(20)
  !$xmp align a(i) with t(i)

  sum = 0
  !$xmp loop on t(i) reduction(+:sum)
  do i=1,20
     a(i) = i
     sum = sum + i
  end do

  if (sum == 210) then
     write(*,*) "PASS"
  end if
  
end subroutine hoge
