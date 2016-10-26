integer a(3)
blkname1 : block
integer a(10)
blkname2 : block
!$xmp nodes p(1)
!$xmp template t(10)
!$xmp distribute t(block) onto p
!$xmp align a(i) with t(i)

!$xmp loop on t(i)
do i=1,10
a(i)=i**2
end do

end block blkname2
if (a(5).eq.25) then
  print *, 'PASS'
else
  print *, 'ERROR'
end if
end block blkname1
end

