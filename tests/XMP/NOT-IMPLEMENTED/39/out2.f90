! testp018.f
! gmove指示文のテスト

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
      integer procs
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
      
      procs = xmp_num_nodes()
      do j=1, procs
         if(j.eq.xmp_node_num()) then
!$xmp gmove out
            a1(:) = a2(:)
!$xmp gmove out
            c1(:) = c2(:)
!$xmp gmove out
            b1(:) = b2(:)
         endif
      enddo

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
      if(sa .ne. 1000 .or.
     $     abs(sb-1000.0) .gt. 0.000000001 .or.
     $     abs(sc-1000.0) .gt. 0.0001) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp018.f ', result
      
      end
