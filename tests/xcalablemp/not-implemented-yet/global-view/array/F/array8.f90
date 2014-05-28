! testp078.f
! task指示文およびarray指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,4)
!$xmp template t(N,N)
!$xmp distribute t(gblock((/100,400,250,250/)),gblock((/100,400,250,250/))) onto p
      integer a(N,N), sa, ansa
      real*8  b(N,N), sb, ansb
      real*4  c(N,N), sc, ansc
      character(len=2) result
!$xmp align a(i,j) with t(i,j)
!$xmp align b(i,j) with t(i,j)
!$xmp align c(i,j) with t(i,j)
!$xmp shadow a(1,1)
!$xmp shadow b(2,2)
!$xmp shadow c(3,3)

      sa = 0
      sb = 0.0
      sc = 0.0
      
!$xmp loop on t(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = 1
            b(i,j) = 1.0
            c(i,j) = 1.0
         enddo
      enddo

!$xmp task on p(1,1)
!$xmp array on t(:,:)
      a(1,1:100) = a(1,1:100)+1
!$xmp array on t(:,:)
      b(1,1:100) = b(1,1:100)+0.5
!$xmp array on t(:,:)
      c(1,1:100) = c(1,1:100)+2.0
!$xmp end task

!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            sa = sa+a(i,j)
            sb = sb+b(i,j)
            sc = sc+c(i,j)
         enddo
      enddo
!$xmp reduction (+: sa, sb, sc)

      result = 'OK'
      ansa = 1000100
      if(sa .ne. ansa) then
         result = 'NG'
      endif
      ansb = 1000050.0
      if(abs(sb-ansb) .gt. 0.0000001) then
         result = 'NG'
      endif
      ansc = 1000200.0
      if(abs(sc-ansc) .gt. 0.00001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp078.f ', result

      end
