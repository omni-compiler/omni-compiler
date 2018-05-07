! testp168.f
! loop指示文とreduction節(-)のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,4,*)
!$xmp template t(N,N,N)
!$xmp distribute t1(cyclic,cyclic,cyclic) onto p
!$xmp distribute t2(cyclic,cyclic(7),cyclic(13)) onto p
!$xmp distribute t3(block,cyclic,cyclic(3)) onto p
      integer a(N,N,N), sa
      real*8  b(N,N,N), sb
      real*4  c(N,N,N), sc
      character(len=2) result
!$xmp align a(i,j,k) with t1(i,j,k)
!$xmp align b(i,j,k) with t2(i,j,k)
!$xmp align c(i,j,k) with t3(i,j,k)

      if(xmp_num_nodes().lt.16) then
         print *, 'You have to run this program by more than 16 nodes.'
      endif

      sa = 0
      sb = 0.0
      sc = 0.0

      a = 1
      b = 0.5
      c = 0.25

!$xmp loop (i,j,k) on t1(i,j,k) reduction(-:sa)
      do k=1, N
         do j=1, N
            do i=1, N
               sa = sa-a(i,j,k)
            enddo
         enddo
      enddo

!$xmp loop (i,j,k) on t2(i,j,k) reduciton(-:sb)
      do k=1, N
         do j=1, N
            do i=1, N
               sb = sb-b(i,j,k)
            enddo
         enddo
      enddo

!$xmp loop (i,j,k) on t3(i,j,k) reduciton(-:sc)
      do k=1, N
         do j=1, N
            do i=1, N
               sc = sc-c(i,j,k)
            enddo
         enddo
      enddo
      
      result = 'OK'
      if(  sa .ne. -N**3 .or.
     $     abs(sb+(dble(N**3)*0.5)) .gt. 0.000001 .or.
     $     abs(sc+(real(N**3)*0.25)) .gt. 0.0001) then
         result = 'NG'
         print *, sa, sb, sc
      endif

      print *, xmp_node_num(), 'testp168.f ', result

      end
