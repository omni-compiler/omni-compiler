subroutine sub1

  REAL, ALLOCATABLE :: intense(:,:)
 
  CALL init_i_j(i, j)

     ALLOCATE (intense(i, j), STAT = ierr4)
     IF (ierr4 == 0) RETURN
     i = i/2; j = j/2

end subroutine sub1

program main
   COMPLEX, POINTER :: hermitian (:, :)
  READ *, m, n
  ALLOCATE (hermitian (m, n))
end program main
