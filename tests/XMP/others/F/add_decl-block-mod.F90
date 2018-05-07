module bm_add_decl_block_mod
  integer a(10)
end module bm_add_decl_block_mod

integer a(10)

!$xmp nodes p(2)

#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
 || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

!$xmp template t(10)
!$xmp distribute t(block) onto p
!$xmp align a(i) with t(i)

!$xmp loop on t(i)
do i=1,10
  a(i)=i**2
end do

!print *, a


blkname1 : block

use bm_add_decl_block_mod

!$xmp nodes p(2)
!$xmp template t(12)
!$xmp distribute t(block) onto p
!$xmp align a(i) with t(i)

!$xmp loop on t(i)
do i=1,10
  a(i)=i**3
end do

!print *, a

!$xmp task on p(1)
if (a(6).eq.216) then
  print *, 'PASS 1'
else
  print *, 'ERROR 1'
  call exit(1)
end if
!$xmp end task

end block blkname1


!print *, a

!$xmp task on p(2)
if (a(6).eq.36) then
  print *, 'PASS 2'
else
  print *, 'ERROR 2'
  call exit(1)
end if
!$xmp end task

#else
!$xmp task on p(2)
  print *, 'SKIPPED'
!$xmp end task
#endif


end

