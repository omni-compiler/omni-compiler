program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(4)
!$xmp template t(N)
!$xmp distribute t(cyclic) onto p
  integer a(N), sa
  real*8  b(N), sb
  real*4  c(N), sc
  integer procs, w, result, w1(4)
!$xmp align a(i) with t(i)
!$xmp align b(i) with t(i)
!$xmp align c(i) with t(i)

  sa = 0
  sb = 0.0
  sc = 0.0

!$xmp loop (i) on t(i)
  do i=1, N
     a(i) = 1
     b(i) = 0.5
     c(i) = 0.01
  end do

!$xmp loop on t(i)
  do i=1, N
     sa = sa+a(i)
  enddo
!$xmp reduction(+:sa) on p(1:2)

!$xmp loop on t(i)
  do i=1, N
     sb = sb+b(i)
  enddo
!$xmp reduction(+:sb) on p(2:3)

!$xmp loop on t(i)
  do i=1, N
     sc = sc+c(i)
  enddo
!$xmp reduction(+:sc) on p(3:4)
  
  procs = xmp_num_nodes()
  w1(:) = N/procs

  result = 0
  if(xmp_node_num().eq.1) then
     if( sa .ne. (w1(1)+w1(2)) .or. abs(sb-(dble(w1(1))*0.5)) .gt. 0.000001 .or. abs(sc-(real(w1(1))*0.01)) .gt. 0.0001) then
        result = -1
     endif
  else if(xmp_node_num().eq.2) then
     if( sa .ne. (w1(1)+w1(2)) .or. abs(sb-(dble(w1(2)+w1(3))*0.5)) .gt. 0.000001 .or. abs(sc-(real(w1(2))*0.01)) .gt. 0.0001) then
        result = -1
     endif
  else if(xmp_node_num().eq.3)then
     if( sa .ne. (w1(3)) .or. abs(sb-(dble(w1(2)+w1(3))*0.5)) .gt. 0.000001 .or. abs(sc-(real(w1(3)+w1(4))*0.01)) .gt. 0.0001) then
        result = -1
     endif
  else if(xmp_node_num().eq.4)then
     if( sa .ne. (w1(4)) .or. abs(sb-(dble(w1(4))*0.5)) .gt. 0.000001 .or. abs(sc-(real(w1(3)+w1(4))*0.01)) .gt. 0.0001) then
        result = -1
     endif
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
