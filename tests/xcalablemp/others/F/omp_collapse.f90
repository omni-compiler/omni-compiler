program foo
  implicit none
  integer :: n,i,k,a(10,10)
  n = 10

  !$OMP PARALLEL DO collapse(2) private(i,k)
  DO i=1,n
     DO k=1,n
        a(k,i) = 1
     ENDDO
  ENDDO
  !$OMP END PARALLEL DO

  !$OMP PARALLEL
  !$OMP DO collapse(2) private(i,k)
  DO i=1,n
     DO k=1,n
        a(k,i) = 1
     ENDDO
  ENDDO
  !$OMP END DO
  !$OMP END PARALLEL
end program foo
