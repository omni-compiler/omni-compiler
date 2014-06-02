! testp009.f
! array指示文のテスト：全体配列+ストライドは1

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
      a(:,:) = 1
!$xmp array on t(:)
      b(:,:) = 2.0
!$xmp array on t(:)
      c(:,:) = 3.0

      result = 'OK'
!$xmp loop on t(j)
      do j=1, N
         do i=1, N
            if(a(i,j).ne.1 .or. b(i,j).ne.2.0 .or. c(i,j).ne.3.0) then
               result = 'NG'
            endif
         enddo
      enddo

      print *, xmp_all_node_num(), 'testp009.f ', result

      end
