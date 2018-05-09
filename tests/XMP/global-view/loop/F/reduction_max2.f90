program main
  include 'xmp_lib.h'
  integer,parameter:: N=10
  integer random_array(N*N), ans_val, result
  integer a(N,N), sa
  real*8  b(N,N), sb
  real*4  c(N,N), sc
  real tmp(N,N)
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(block,block) onto p
  
  result = 0
  call random_number(tmp)
  do j=1, N
     do i=1, N
        l = (j-1)*N+i
        random_array(l) = int(tmp(i,j) * 10000)
     end do
  end do

!$xmp loop (i,j) on t(i,j)
  do j=1, N
     do i=1, N
        l = (j-1)*N+i
        a(i,j) = random_array(l)
        b(i,j) = dble(random_array(l))
        c(i,j) = real(random_array(l))
     enddo
  enddo
         
  ans_val = 0
  do j=1, N
     do i=1, N
        l = (j-1)*N+i
        ans_val = max(ans_val, random_array(l))
     enddo
  enddo
  
  sa = 0
  sb = 0.0
  sc = 0.0

!$xmp loop (i,j) on t(i,j) reduction(max: sa, sb, sc)
  do j=1, N
     do i=1, N
        sa = max(sa, a(i,j))
        sb = max(sb, b(i,j))
        sc = max(sc, c(i,j))
     enddo
  enddo

  if( sa .ne. ans_val .or. sb .ne. dble(ans_val) .or. sc .ne. real(ans_val) ) then
     result = -1
  endif

!$xmp reduction(+:result)
!$xmp task on p(1,1)
  if( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task
      
end program main
