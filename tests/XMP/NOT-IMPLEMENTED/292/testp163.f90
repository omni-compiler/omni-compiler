program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
  integer random_array(N), ans_val
!$xmp nodes p(*)
!$xmp template t1(N,N,N)
!$xmp template t2(N,N,N)
!$xmp template t3(N,N,N)
!$xmp distribute t1(block,*,*) onto p
!$xmp distribute t2(*,block,*) onto p
!$xmp distribute t3(*,*,block) onto p
  integer a(N), sa
  real*8  b(N), sb
  real*4  c(N), sc
  integer ia, ib, ic, ii, result
  real tmp(N)
!$xmp align a(i) with t1(i,*,*)
!$xmp align b(i) with t2(*,i,*)
!$xmp align c(i) with t3(*,*,i)
  
  result = 0
  call random_number(tmp)
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

  ans_val = 0
  ii = 1
  do i=1, N
     if(ans_val .lt. random_array(i)) then
        ii = i
        ans_val = random_array(i)
     endif
  enddo

  sa = 0
  sb = 0.0
  sc = 0.0
  ia = 1
  ib = 1
  ic = 1

!$xmp loop (i) on t1(i,:,:) reduction(firstmax:sa/ia/)
  do i=1, N
     if(sa .lt. a(i)) then
        ia = i
        sa = a(i)
     endif
  enddo

!$xmp loop (i) on t2(:,i,:) reduction(firstmax:sb/ib/)
  do i=1, N
     if(sb .lt. b(i)) then
        ib = i
        sb = b(i)
     endif
  enddo
  
!$xmp loop (i) on t3(:,:,i) reduction(firstmax:sc/ic/)
  do i=1, N
     if(sc .lt. c(i)) then
        ic = i
        sc = c(i)
     endif
  enddo

  if( sa .ne. ans_val .or. sb .ne. dble(ans_val) .or. sc .ne. real(ans_val) .or. ia .ne. ii .or. ib .ne. ii .or. ic .ne. ii) then
     result = -1
  endif

end program main
