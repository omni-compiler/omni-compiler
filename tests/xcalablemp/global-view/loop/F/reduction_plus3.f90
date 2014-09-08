program main
  include 'xmp_lib.h'
  integer,parameter:: N=100
!$xmp nodes p(4,4,*)
!$xmp template t1(N,N,N)
!$xmp template t2(N,N,N)
!$xmp template t3(N,N,N)
!$xmp distribute t1(block,block,block) onto p
!$xmp distribute t2(block,block,cyclic) onto p
!$xmp distribute t3(block,cyclic,cyclic) onto p
  integer a(N,N,N), sa, result
  real*8  b(N,N,N), sb
  real*4  c(N,N,N), sc
!$xmp align a(i,j,k) with t1(i,j,k)
!$xmp align b(i,j,k) with t2(i,j,k)
!$xmp align c(i,j,k) with t3(i,j,k)

  sa = 0
  sb = 0.0
  sc = 0.0

!$xmp loop (i,j,k) on t1(i,j,k)
  do k=1, N
     do j=1, N
        do i=1, N
           a(i,j,k) = 1
        enddo
     enddo
  enddo

!$xmp loop (i,j,k) on t2(i,j,k)
  do k=1, N
     do j=1, N
        do i=1, N
           b(i,j,k) = 0.5
        enddo
     enddo
  enddo

!$xmp loop (i,j,k) on t3(i,j,k)
  do k=1, N
     do j=1, N
        do i=1, N
           c(i,j,k) = 0.25
        enddo
     enddo
  enddo
  
!$xmp loop (i,j,k) on t1(i,j,k) reduction(+:sa)
  do k=1, N
     do j=1, N
        do i=1, N
           sa = sa+a(i,j,k)
        enddo
     enddo
  enddo

!$xmp loop (i,j,k) on t2(i,j,k) reduction(+:sb)
  do k=1, N
     do j=1, N
        do i=1, N
           sb = sb+b(i,j,k)
        enddo
     enddo
  enddo
  
!$xmp loop (i,j,k) on t3(i,j,k) reduction(+:sc)
  do k=1, N
     do j=1, N
        do i=1, N
           sc = sc+c(i,j,k)
        enddo
     enddo
  enddo
  
  result = 0
  if( sa .ne. N**3 .or. abs(sb-(dble(N**3)*0.5)) .gt. 0.000001 .or. abs(sc-(real(N**3)*0.25)) .gt. 0.0001) then
     result = -1
  endif

!$xmp reduction(+:result)
!$xmp task on p(1,1,1)
  if( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program main
      
