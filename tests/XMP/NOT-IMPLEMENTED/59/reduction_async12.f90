! testp049.f
! reduction指示文(*)のテスト

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

      sa = 1
      sb = 1.0
      sc = 1.0

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
            m = (j-1)*N+i
            if(mod(m,4).eq.0) then
               b(i,j) = 1.0
            else if(mod(m,4).eq.1) then
               b(i,j) = 2.0
            else if(mod(m,4).eq.2) then
               b(i,j) = 4.0
            else
               b(i,j) = 0.125
            endif
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            m = (j-1)*N+i
            if(mod(m,4).eq.0) then
               c(i,j) = 0.5
            else if(mod(m,4).eq.1) then
               c(i,j) = 2.0
            else if(mod(m,4).eq.2) then
               c(i,j) = 4.0
            else
               c(i,j) = 0.25
            endif
         enddo
      enddo

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            sa = sa*a(i,j)
         enddo
      enddo
!$xmp reduction (*:sa) async(1)

!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            sb = sb*b(i,j)
         enddo
      enddo
!$xmp reduction (*:sb) async(2)

!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            sc = sc*c(i,j)
         enddo
      enddo
!$xmp reduction (*:sc) async(3)
!$xmp wait_async(1)
      sa = sa/2
!$xmp wait_async(2)
      sb = sb/2.0
!$xmp wait_async(3)
      sc = sc/4.0
      
      result = 'OK'
      if(  sa .ne. 512 .or.
     $     abs(sb-0.5) .gt. 0.000001 .or.
     $     abs(sc-0.25) .gt. 0.0001) then
         result = 'NG'
         print *, sa, sb, sc
      endif

      print *, xmp_node_num(), 'testp049.f ', result

      end
