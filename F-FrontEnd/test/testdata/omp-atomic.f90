  program main

    real r = 0

!$omp parallel do
    do i = 1, 10
!$omp atomic
       r = r + 1
!$omp barrier
    end do
!$omp end parallel do

    print *, d

  end program main
