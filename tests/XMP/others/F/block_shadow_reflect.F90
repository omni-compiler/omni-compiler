!$xmp nodes p(4)

#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
 || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

!$xmp template t(12)
!$xmp distribute t(block) onto p
integer ary1(12)
integer ary2(12)
!$xmp align ary1(i) with t(i)
!$xmp shadow ary1(1)

!$xmp loop on t(i)
do i=1,12
  ary1(i) = i
enddo

!print *, 'line 01 : ', ary1

!$xmp reflect (ary1)

!print *, 'line 02 : ', ary1

!$xmp loop on t(i)
do i=1,12
  ary2(i) = ary1(i-1) + ary1(i+1)
enddo

!$xmp loop on t(i)
do i=1,12
  ary1(i) = ary2(i)
enddo

!print *, 'line 03 : ', ary1

!$xmp task on p(1)
if (ary1(3).eq.6) then
  print *, 'PASS 1'
else
  print *, 'ERROR 1'
  call exit(1)
end if
!$xmp end task


blkname1 : block
!$xmp nodes p(4)
!$xmp template t(13)
!$xmp distribute t(block) onto p
integer ary1(12)
integer ary2(12)
!$xmp align ary1(i) with t(i)
!$xmp shadow ary1(1)

!$xmp loop on t(i)
do i=1,12
  ary1(i) = i
enddo

!print *, 'line 04 : ', ary1

!$xmp reflect (ary1)

!print *, 'line 05 : ', ary1

!$xmp loop on t(i)
do i=1,12
  ary2(i) = ary1(i-1) + ary1(i) + ary1(i+1)
enddo

!$xmp loop on t(i)
do i=1,12
  ary1(i) = ary2(i)
enddo

!print *, 'line 06 : ', ary1


!$xmp task on p(2)
if (ary1(7).eq.21) then
  print *, 'PASS 2'
else
  print *, 'ERROR 2'
  call exit(1)
end if
!$xmp end task


end block blkname1

#else
!$xmp task on p(2)
  print *, 'SKIPPED'
!$xmp end task
#endif

end

