! testp011.f
! array指示文のテスト：部分配列+ストライドあり

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer a(N,N)
      real*4 b(N,N)
      real*8 c(N,N)
      character(len=2) result
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
!$xmp align a(*,i) with t(i)
!$xmp align b(*,i) with t(i)
!$xmp align c(*,i) with t(i)
      common /xxx/ a, c

!$xmp loop on t(j)
      do j=1, N
         do i=1, N
            a(i,j) = (j-1)*N+i
            b(i,j) = real((j-1)*N+i)
            c(i,j) = dble((j-1)*N+i)
         enddo
      enddo

!$xmp array on t(:)
      a(222:777:2,333:444:3) = 0
!$xmp array on t(:)
      b(111:777:3,444:555:4) = 0.0
!$xmp array on t(:)
      c(222:888:4,555:666:5) = 0.0

      result = 'OK'

!$xmp loop on t(j)
      do j=333,444,3
         do i=222,777,2
            if(a(i,j) .ne. 0) then
               result = 'NG'
            else
               a(i,j) = (j-1)*N+i
            endif
         enddo
      enddo

!$xmp loop on t(j)
      do j=444,555,4
         do i=111,777,3
            if(b(i,j) .ne. 0.0) then
               result = 'NG'
            else
               b(i,j) = real((j-1)*N+i)
            endif
         enddo
      enddo

!$xmp loop on t(j)
      do j=555,666,5
         do i=222,888,4
            if(c(i,j) .ne. 0.0) then
               result = 'NG'
            else
               c(i,j) = dble((j-1)*N+i)
            endif
         enddo
      enddo

!$xmp loop on t(j)
      do j=1, N
         do i=1, N
            if(a(i,j) .ne. (j-1)*N+i) result = 'NG'
            if(b(i,j) .ne. real((j-1)*N+i)) result = 'NG'
            if(c(i,j) .ne. dble((j-1)*N+i)) result = 'NG'
         enddo
      enddo

      print *, xmp_all_node_num(), 'testp011.f ', result

      end
