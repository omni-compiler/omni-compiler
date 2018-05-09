program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t1(N,N)
!$xmp template t2(N,N)
!$xmp template t3(N,N)
!$xmp distribute t1(block,block) onto p
!$xmp distribute t2(cyclic,block) onto p
!$xmp distribute t3(cyclic,cyclic(5)) onto p
  integer a(N,N), sa, result
  real*8  b(N,N), sb
  real*4  c(N,N), sc
!$xmp align a(i,j) with t1(i,j)
!$xmp align b(i,j) with t2(i,j)
!$xmp align c(i,j) with t3(i,j)

!$xmp loop (i,j) on t1(i,j)
  do j=1, N
     do i=1, N
        if(i.eq.j .and. mod(i,100).eq.0) then
           a(i,j) = 2
        else
           a(i,j) = 1
        endif
     enddo
  enddo
  
!$xmp loop (i,j) on t2(i,j)
  do j=1, N
     do i=1, N
        if(mod(j,2).eq.0) then
           if(mod(i,2).eq.0) then
              b(i,j) = 2.0
           else
              b(i,j) = 1.0
           endif
        else
           if(mod(i,2).eq.1) then
              b(i,j) = 0.5
           else
              b(i,j) = 1.0
           endif
        endif
     enddo
  enddo

!$xmp loop (i,j) on t3(i,j)
  do j=1, N
     do i=1, N
        if(mod(j,2).eq.0) then
           if(mod(i,4).eq.0) then
              c(i,j) = 1.0
           else if(mod(i,4).eq.1) then
              c(i,j) = 4.0
           else if(mod(i,4).eq.2) then
              c(i,j) = 1.0
           else
              c(i,j) = 0.25
           endif
        else
           if(mod(i,4).eq.0) then
              c(i,j) = 0.25
           else if(mod(i,4).eq.1) then
              c(i,j) = 1.0
           else if(mod(i,4).eq.2) then
              c(i,j) = 4.0
           else
              c(i,j) = 1.0
           endif
        endif
     enddo
  enddo
  
  sa = 1
  sb = 1.0
  sc = 1.0
  
!$xmp loop (i,j) on t1(i,j) reduction(*:sa)
  do j=1, N
     do i=1, N
        sa = sa*a(i,j)
     enddo
  enddo
  
!$xmp loop (i,j) on t2(i,j) reduction(*:sb)
  do j=1, N
     do i=1, N
        sb = sb*b(i,j)
     enddo
  enddo
  
!$xmp loop (i,j) on t3(i,j) reduction(*:sc)
  do j=1, N
     do i=1, N
        sc = sc*c(i,j)
     enddo
  enddo
      
  result = 0
  if( sa .ne. 1024 .or. abs(sb-1.0) .gt. 0.000001 .or.  abs(sc-1.0) .gt. 0.0001) then
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
