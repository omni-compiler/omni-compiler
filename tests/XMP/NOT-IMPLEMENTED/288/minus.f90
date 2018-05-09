program main
  include 'xmp_lib.h'
  integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
  integer a(N), sa
  real*8  b(N), sb
  real*4  c(N), sc
  integer procs, w, remain
  integer,allocatable:: w1(:)
  character(len=2) result
!$xmp align a(i) with t(i)
!$xmp align b(i) with t(i)
!$xmp align c(i) with t(i)

  if(xmp_num_nodes().lt.4) then
     print *, 'You have to run this program by more than 4 nodes.'
  endif

  sa = 0
  sb = 0.0
  sc = 0.0
  
!$xmp loop on t(i)
  do i=1, N
     a(i) = 1
     b(i) = 0.5
     c(i) = 0.01
  enddo
  
!$xmp loop on t(i)
  do i=1, N
     sa = sa-a(i)
  enddo
!$xmp reduction(-:sa) on p(1:2)

!$xmp loop on t(i)
  do i=1, N
     sb = sb-b(i)
  enddo
!$xmp reduction(-:sb) on p(2:3)

!$xmp loop on t(i)
  do i=1, N
     sc = sc-c(i)
  enddo
!$xmp reduction(-:sc) on p(3:4)

  procs = xmp_num_nodes()
  if(mod(N,procs).eq.0) then
     w = N/procs
  else
     w = N/procs+1
  endif
  allocate(w1(1:procs))
  remain = N
  do i=1, procs-1
     w1(i) = min(w, remain)
     remain = remain-w1(i)
  enddo
  w1(procs) = remain
      
  result = 'OK'
  if(xmp_node_num().eq.1)then
     if( sa .ne. -((w1(1)+w1(2)+1000)) .or. abs(sb+(dble(w1(1))*0.5+30.0)) .gt. 0.000001 .or. abs(sc+(real(w1(1))*0.01+25.0)) .gt. 0.000001) then
        result = 'NG'
     endif
  else if(xmp_node_num().eq.2)then
     if( sa .ne. -(w1(1)+w1(2)+1000) .or. abs(sb+(dble(w1(2)+w1(3))*0.5+30.0)) .gt. 0.000001 .or. abs(sc+(real(w1(2))*0.01+25.0)) .gt. 0.000001) then
        result = 'NG'
     endif
  else if(xmp_node_num().eq.3)then
     if( sa .ne. -(w1(3)+1000) .or. abs(sb+(dble(w1(2)+w1(3))*0.5+30.0)) .gt. 0.000001 .or. abs(sc+(real(w1(3)+w1(4))*0.01+25.0)) .gt. 0.000001) then
        result = 'NG'
     endif
  else if(xmp_node_num().eq.4)then
     if( sa .ne. -(w1(4)+1000) .or. abs(sb+(dble(w1(4))*0.5+30.0)) .gt. 0.000001 .or. abs(sc+(real(w1(3)+w1(4))*0.01+25.0)) .gt. 0.000001) then
        result = 'NG'
     endif
  else
     i = xmp_node_num()
     if( sa .ne. -(w1(i)+1000) .or. abs(sb+(dble(w1(i))*0.5+30.0)) .gt. 0.000001 .or. abs(sc+(real(w1(i))*0.01+25.0)) .gt. 0.000001) then
        result = 'NG'
     endif
  endif

  print *, xmp_node_num(), 'testp032.f ', result
  deallocate(w1)
  
      end
      
