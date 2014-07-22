! testp010.f
! array指示文のテスト：部分配列+ストライドは1

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer a(N,N)
      real*4 b(N,N)
      real*8 c(N,N)
      character(len=2) result
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(cyclic(2)) onto p
!$xmp align a(*,i) with t(i)
!$xmp align b(*,i) with t(i)
!$xmp align c(*,i) with t(i)
      common /xxx/ a, b, c

!$xmp loop on t(j)
      do j=1, N
         do i=1, N
            a(i,j) = (j-1)*N+i
            b(i,j) = real((j-1)*N+i)
            c(i,j) = dble((j-1)*N+i)
         enddo
      enddo

!$xmp array on t(:)
      a(222:777,333:444) = 0
!$xmp array on t(:)
      b(111:777,444:555) = 0.0
!$xmp array on t(:)
      c(222:888,555:666) = 0.0

      result = 'OK'
!$xmp loop on t(j)
      do j=1, N
         do i=1, N
            if(i.ge.222.and.i.le.777.and.j.ge.333.and.j.le.444) then
               if(a(i,j) .ne. 0) result = 'NG'
            else
               if(a(i,j) .ne. (j-1)*N+i) result = 'NG'
            endif
            if(i.ge.111.and.i.le.777.and.j.ge.444.and.j.le.555) then
               if(b(i,j) .ne. 0.0) result = 'NG'
            else
               if(b(i,j) .ne. real((j-1)*N+i)) result = 'NG'
            endif
            if(i.ge.222.and.i.le.888.and.j.ge.555.and.j.le.666) then
               if(c(i,j) .ne. 0.0) result = 'NG'
            else
               if(c(i,j) .ne. dble((j-1)*N+i)) result = 'NG'
            endif
         enddo
      enddo

      print *, xmp_node_num(), 'testp010.f ', result

      end
