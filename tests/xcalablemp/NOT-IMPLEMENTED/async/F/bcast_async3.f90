! testp097.f
! loop指示文とbcast指示文のテスト (ノードが1次元の場合、意味なし？)

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t1(N)
!$xmp template t2(N)
!$xmp template t3(N)
!$xmp distribute t1(cyclic(3)) onto p
!$xmp distribute t2(block)) onto p
!$xmp distribute t3(cyclic) onto p
      integer a(N), aa
      real*8  b(N), bb
      real*4  c(N), cc
!$xmp align a(i) with t1(i)
!$xmp align b(i) with t2(i)
!$xmp align c(i) with t3(i)
      character(len=3) result

!$xmp loop on t1(i)
      do i=1, N
         a(i) = -1
      enddo
!$xmp loop on t2(i)
      do i=1, N
         b(i) = -2.0
      enddo
!$xmp loop on t3(i)
      do i=1, N
         c(i) = -3.0
      enddo

      result = 'OK '
!$xmp loop on t1(i)
      do i=1, N
         aa = i
!$xmp bcast (aa) async(1)
         if(a(i) .ne. -1) result = 'NG1'
!$xmp wait_async(1)
         a(i) = aa
      enddo
!$xmp loop on t2(i)
      do i=1, N
         bb = dble(i)
!$xmp bcast (bb) async(2)
         if(b(i) .ne. -2.0) result = 'NG2'
!$xmp wait_async(2)
         b(i) = bb
      enddo
!$xmp loop on t3(i)
      do i=1, N
         cc = real(i)
!$xmp bcast (cc) async(3)
         if(c(i) .ne. -3.0) result = 'NG3'
!$xmp wait_async(3)
         c(i) = cc
      enddo

!$xmp loop on t1(i)
      do i=1, N
         if(a(i) .ne. i) result = 'NG4'
      enddo
!$xmp loop on t2(i)
      do i=1, N
         if(abs(b(i)-dble(i)) .gt. 0.00000001) result = 'NG5'
      enddo
!$xmp loop on t3(i)
      do i=1, N
         if(abs(c(i)-real(i)) .gt. 0.00000001) result = 'NG6'
      enddo

      print *, xmp_node_num(), 'testp097.f ', result

      end
      
