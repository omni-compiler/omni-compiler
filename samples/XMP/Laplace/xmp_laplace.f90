program laplace
  integer,parameter :: N1=64, N2=64
  real(8),parameter :: PI = 3.141592653589793238463

  integer :: niter=100
  real(8) :: u(N1,N2),uu(N1,N2)
  real(8) :: value = 0.0

!$xmp nodes p(4,*)
!$xmp template t(N1,N2)
!$xmp distribute t(block, block) onto p
!$xmp align u(i,j) with t(i, j)
!$xmp align uu(i,j) with t(i, j)
!$xmp shadow uu(1:1,1:1)

!$xmp loop (i,j) on t(i,j)
  do j=1,N2
     do i=1,N1
        u(i,j)=0.0
        uu(i,j)=0.0
     end do
  end do

!$xmp loop (i,j) on t(i,j)
  do j=2,N2-1
     do i=2,N1-1
        u(i,j)=sin(dble(i-1)/N1*PI)+cos(dble(j-1)/N2*PI)
     end do
  end do

  do k=1,niter

!$xmp loop (i,j) on t(i,j)
     do j=2,N2-1
        do i=2,N1-1
           uu(i,j)=u(i,j)
        end do
     end do

!$xmp reflect (uu)

!$xmp loop (i,j) on t(i,j)
     do j=2,N2-1
        do i=2,N1-1
           u(i,j)=(uu(i-1,j) + uu(i+1,j) + uu(i,j-1) + uu(i,j+1))/4.0
        end do
     end do

  enddo

!$xmp loop (i,j) on t(i,j) reduction(+:value)
  do j=2,N2-1
     do i=2,N1-1
        value = value + dabs(uu(i,j) -u(i,j))
     end do
  end do

!$xmp task on p(1,1)
  print *, 'Verification =', value
!$xmp end task
  
end program laplace
