! testp047.f
! reduction指示文(+)のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t1(block,block) onto p
!$xmp distribute t2(block,cyclic) onto p
!$xmp distribute t3(cyclic,cyclic) onto p
      integer a(N,N), sa
      real*8  b(N,N), sb
      real*4  c(N,N), sc
      character(len=2) result
!$xmp align a(i,j) with t1(i,j)
!$xmp align b(i,j) with t2(i,j)
!$xmp align c(i,j) with t3(i,j)

      if(xmp_num_nodes().lt.4) then
         print *, 'You have to run this program by more than 4 nodes.'
      endif

      sa = 0
      sb = 0.0
      sc = 0.0

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = 1
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            b(i,j) = 0.5
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            c(i,j) = 0.25
         enddo
      enddo

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            sa = sa+a(i,j)
         enddo
      enddo
!$xmp reduction (+:sa) async(1)

!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            sb = sb+b(i,j)
         enddo
      enddo
!$xmp reduction (+:sb) async(2)

!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp reduction (+:sc) async(3)
!$xmp wait_async(1)
      sa = sa+10
!$xmp wait_async(2)
      sb = sb+20.0
!$xmp wait_async(3)
      sc = sc+30.0
      
      result = 'OK'
      if(  sa .ne. N**2+10 .or.
     $     abs(sb-(dble(N**2)*0.5+20.0)) .gt. 0.000001 .or.
     $     abs(sc-(real(N**2)*0.25+30.0)) .gt. 0.0001) then
         result = 'NG'
         print *, sa, sb, sc
      endif

      print *, xmp_node_num(), 'testp047.f ', result

      end
