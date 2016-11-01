!$xmp nodes p(2)

#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
 || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

integer :: ary1(6) = (/1,2,3,4,5,6/)
integer :: ary2(10) = (/1,2,3,4,5,6,7,8,9,10/)
!print *, 'line 01 : ', ary1
!print *, 'line 02 : ', ary2


blkname1 : block

integer :: ary1(10)
integer :: ary2(10)

!$xmp nodes p(2)
!$xmp template t(10)
!$xmp distribute t(block) onto p
!$xmp align ary1(i) with t(i)

!$xmp loop on t(i)
do i=1,10
  ary1(i)=i**2
end do
!print *, 'line 03 : ', ary1


blkname2 : block

integer :: ary1(10)
integer :: ary2(10)
!$xmp nodes p(2)
!$xmp template t(12)
!$xmp distribute t(block) onto p
!$xmp align ary1(i) with t(i)

!$xmp loop on t(i)
do i=1,10
  ary1(i)=i**3
end do
!print *, 'line 04 : ', ary1

!$xmp task on p(1)
if (ary1(6).eq.216) then
  print *, 'PASS 1'
else
  print *, 'ERROR 1'
  call exit(1)
end if
!$xmp end task

!$xmp gmove
ary2(:)=ary1(:)
!print *, 'line 05 : ', ary2

!!$xmp task on p(2)
if (ary2(6).eq.216) then
  print *, 'PASS 2'
else
  print *, 'ERROR 2'
  call exit(1)
end if
!!$xmp end task

end block blkname2


!$xmp gmove
ary2(:)=ary1(:)
!print *, 'line 06 : ', ary2


blkname3 : block

integer :: ary1(10)
integer :: ary2(10)
!$xmp nodes p(2)
!$xmp template t(14)
!$xmp distribute t(block) onto p
!$xmp align ary1(i) with t(i)

!$xmp loop on t(i)
do i=1,10
  ary1(i)=i**4
end do
!print *, 'line 07 : ', ary1

!$xmp task on p(1)
if (ary1(7).eq.2401) then
  print *, 'PASS 3'
else
  print *, 'ERROR 3'
  call exit(1)
end if
!$xmp end task

!$xmp gmove
ary2(:)=ary1(:)
!print *, 'line 08 : ', ary2

!$xmp task on p(2)
if (ary2(7).eq.2401) then
  print *, 'PASS 4'
else
  print *, 'ERROR 4'
  call exit(1)
end if
!$xmp end task

end block blkname3


!print *, 'line 09 : ', ary1
!print *, 'line 10 : ', ary2

!$xmp task on p(1)
if (ary1(5).eq.25) then
  print *, 'PASS 5'
else
  print *, 'ERROR 5'
  call exit(1)
end if
!$xmp end task

!$xmp task on p(2)
if (ary2(5).eq.25) then
  print *, 'PASS 6'
else
  print *, 'ERROR 6'
  call exit(1)
end if
!$xmp end task

end block blkname1


!print *, 'line 11 : ', ary1
!print *, 'line 12 : ', ary2

!$xmp task on p(1)
if (ary1(6).eq.6.and.ary2(10).eq.10) then
  print *, 'PASS 7'
else
  print *, 'ERROR 7'
  call exit(1)
end if
!$xmp end task

#else
!$xmp task on p(1)
  print *, 'SKIPPED'
!$xmp end task
#endif

end

