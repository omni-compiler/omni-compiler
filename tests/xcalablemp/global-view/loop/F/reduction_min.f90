program main
  include 'xmp_lib.h'
  integer,parameter:: N=10
  integer random_array(N), ans_val, result
!$xmp nodes p(*)
!$xmp template t1(N,N,N)
!$xmp template t2(N,N,N)
!$xmp template t3(N,N,N)
!$xmp distribute t1(cyclic,*,*) onto p
!$xmp distribute t2(*,cyclic(2),*) onto p
!$xmp distribute t3(*,*,cyclic(7)) onto p
  integer a(N), sa
  real*8  b(N), sb
  real*4  c(N), sc
  real tmp(N)
!$xmp align a(i) with t1(i,*,*)
!$xmp align b(i) with t2(*,i,*)
!$xmp align c(i) with t3(*,*,i)

  result = 0

  call random_number( tmp )
  random_array(:) = int(tmp(:) * 10000)

!$xmp loop (i) on t1(i,:,:)
  do i=1, N
     a(i) = random_array(i)
  enddo

!$xmp loop (i) on t2(:,i,:)
  do i=1, N
     b(i) = dble(random_array(i))
  enddo

!$xmp loop (i) on t3(:,:,i)
  do i=1, N
     c(i) = real(random_array(i))
  enddo

  ans_val = 10000
  do i=1, N
     ans_val = min(ans_val, random_array(i))
  enddo

  sa = 10000
  sb = 10000.0
  sc = 10000.0

!$xmp loop (i) on t1(i,:,:) reduction(min: sa)
  do i=1, N
     sa = min(sa, a(i))
  enddo

!$xmp loop (i) on t2(:,i,:) reduction(min: sb)
  do i=1, N
     sb = min(sb, b(i))
  enddo

!$xmp loop (i) on t3(:,:,i) reduction(min: sc)
  do i=1, N
     sc = min(sc, c(i))
  enddo

  if( sa .ne. ans_val .or. sb .ne. dble(ans_val) .or. sc .ne. real(ans_val) ) then
     result = -1
  endif

!$xmp reduction(+:result)
!$xmp task on p(1)
  if( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task

end program main
