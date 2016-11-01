program main
  integer :: k
!$xmp nodes p(4)
#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
 || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
block
#endif
  integer, save :: k
!$omp threadprivate (k)
  call sub
#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
 || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
end block
#endif
!$xmp task on p(1)
  print *, 'PASS'
!$xmp end task
end program main

subroutine sub
  integer, save :: i, j
!$omp threadprivate (i,j)
  k = 1
end subroutine sub
