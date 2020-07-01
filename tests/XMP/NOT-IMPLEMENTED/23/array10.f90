! testp100.f
! loop指示文とarray指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t1(N,N)
!$xmp template t2(N,N)
!$xmp template t3(N,N)
!$xmp distribute t1(cyclic(3),cyclic(7)) onto p
!$xmp distribute t2(block,cyclic) onto p
!$xmp distribute t3(cyclic,block) onto p
      integer a1(N,N), a2(N,N)
      real*8  b1(N,N), b2(N,N)
      real*4  c1(N,N), c2(N,N)
!$xmp align a1(i,j) with t1(i,j)
!$xmp align a2(i,j) with t1(i,j)
!$xmp align b1(i,j) with t2(i,j)
!$xmp align b2(i,j) with t2(i,j)
!$xmp align c1(i,j) with t3(i,j)
!$xmp align c2(i,j) with t3(i,j)
      character(len=2) result

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            a2(i,j) = (j-1)*N+i
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            b2(i,j) = dble((j-1)*N+i)
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            c2(i,j) = real((j-1)*N+i)
         enddo
      enddo

!$xmp loop (j) on t1(:,j)
      do j=1, N
!$xmp array on t1(i,j)
         a1(:,j) = a2(:,j)+1
      enddo
!$xmp loop (j) on t2(:,j)
      do j=1, N
!$xmp array on t2(i,j)
         b1(:,j) = b2(:,j)+2.0
      enddo
!$xmp loop (j) on t3(:,j)
      do j=1, N
!$xmp array on t3(i,j)
         c1(:,j) = c2(:,j)+3.0
      enddo

      result = 'OK'
!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            if(a1(i,j) .ne. (j-1)*N+i+1) result = 'NG'
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            if(abs(b1(i,j)-dble((j-1)*N+i+2)).gt.0.0000001) result='NG'
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            if(abs(c1(i,j)-dble((j-1)*N+i+3)).gt.0.001) result = 'NG'
         enddo
      enddo

      print *, xmp_node_num(), 'testp100.f ', result

      end
