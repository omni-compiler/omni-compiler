! testp012.f
! array指示文のテスト：多次元分散 + 全体配列

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer a(N,N)
      real*4 b(N,N)
      real*8 c(N,N)
      character(len=2) result
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(cyclic(2), cyclic(3)) onto p
!$xmp align a(i,j) with t(i,j)
!$xmp align b(i,j) with t(i,j)
!$xmp align c(i,j) with t(i,j)

!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = (j-1)*N+i-1
            b(i,j) = real((j-1)*N+i-1)
            c(i,j) = dble((j-1)*N+i-1)
         enddo
      enddo

!$xmp array on t(:,:)
      a(:,:) = a(:,:)+1
!$xmp array on t(:,:)
      b(:,:) = b(:,:)+1.0
!$xmp array on t(:,:)
      c(:,:) = c(:,:)+1.0
      
      result = 'OK'
!$xmp loop (j) on t(:,j)
      do j=1, N
!$xmp loop (i) on t(i,j)
         do i=1, N
            if(a(i,j) .ne. (j-1)*N+i) result = 'NG'
            if(b(i,j) .ne. real((j-1)*N+i)) result = 'NG'
            if(c(i,j) .ne. dble((j-1)*N+i)) result = 'NG'
         enddo
      enddo

      print *, xmp_node_num(), 'testp012.f ', result

      end
