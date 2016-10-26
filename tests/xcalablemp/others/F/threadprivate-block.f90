program main
!$xmp nodes p(4)
block
  integer, save :: k
!$omp threadprivate (k)
  call sub
end block
end program main

subroutine sub
  integer, save :: i, j
!$omp threadprivate (i,j)
  k = 1
end subroutine sub
