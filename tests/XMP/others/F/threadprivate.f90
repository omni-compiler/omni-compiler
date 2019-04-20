program main
  integer, save :: k
!$xmp nodes p(4)
!$omp threadprivate (k)
  call sub
end program main

subroutine sub
  integer, save :: i, j
!$omp threadprivate (i,j)
  k = 1
end subroutine sub
