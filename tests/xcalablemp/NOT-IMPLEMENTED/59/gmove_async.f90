! testp019.f
! gmove指示文とasync節のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t1(N,N)
!$xmp template t2(N,N)
!$xmp distribute t1(*,block) onto p
!$xmp distribute t2(cyclic,*) onto p
      integer a1(N), a2(N), sa
      real*8  b1(N), b2(N), sb
      real*4  c1(N), c2(N), sc
!$xmp align a1(i) with t1(*,i)
!$xmp align b1(i) with t1(*,i)
!$xmp align c1(i) with t1(*,i)
!$xmp align a2(i) with t2(i,*)
!$xmp align b2(i) with t2(i,*)
!$xmp align c2(i) with t2(i,*)
      character(len=2) result

!$xmp loop on t1(:,i)
      do i=1, N
         a1(i) = i
         b1(i) = dble(i)
         c1(i) = real(i)
      enddo

!$xmp looop on t2(i,:)
      do i=1, N
         a2(i) = 1
         b2(i) = 1.0
         c2(i) = 1.0
      enddo
      
!$xmp gmove async(1)
      a1(:) = a2(:)
!$xmp gmove async(3)
      c1(:) = c2(:)
!$xmp gmove async(2)
      b1(:) = b2(:)

!$xmp wait_async(1)
      if(xmp_all_node_num() .eq. 1) then
         a1(1) = a1(1)+10
      endif
!$xmp wait_async(2)
      if(xmp_all_node_num() .eq. 1) then
         b1(1) = b1(1)+30.0
      endif
!$xmp wait_async(3)
      if(xmp_all_node_num() .eq. 1) then
         c1(1) = c1(1)+20.0
      endif

      sa = 0
      sb = 0.0
      sc = 0.0
!$xmp loop on t1(:,i) reduction(+:sa,sb,sc)
      do i=1, N
         sa = sa+a1(i)
         sb = sb+b1(i)
         sc = sc+c1(i)
      enddo

      result = 'OK'
      if(sa .ne. 1010 .or.
     $     abs(sb-1030.0) .gt. 0.000000001 .or.
     $     abs(sc-1020.0) .gt. 0.0001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp019.f ', result
      
      end
