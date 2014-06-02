! testp093.f
! loop指示文とgmove指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t1(N)
!$xmp template t2(N)
!$xmp template t3(N)
!$xmp template t4(N)
!$xmp template t5(N)
!$xmp template t6(N)
!$xmp distribute t1(block) onto p
!$xmp distribute t2(cyclic) onto p
!$xmp distribute t3(cyclic(2)) onto p
!$xmp distribute t4(cyclic(3)) onto p
!$xmp distribute t5(cyclic(5)) onto p
!$xmp distribute t6(cyclic(7)) onto p
      integer a1(N), a2(N)
      real*8  b1(N), b2(N)
      real*4  c1(N), c2(N)
!$xmp align a1(i) with t1(i)
!$xmp align a2(i) with t2(i)
!$xmp align b1(i) with t3(i)
!$xmp align b2(i) with t4(i)
!$xmp align c1(i) with t5(i)
!$xmp align c2(i) with t6(i)
      character(len=3) result

!$xmp loop on t1(i)
      do i=1, N
         a1(i) = 0
      enddo
!$xmp loop on t3(i)
      do i=1, N
         b1(i) = 0.0
      enddo
!$xmp loop on t5(i)
      do i=1, N
         c1(i) = 0.0
      enddo

!$xmp loop on t2(i)
      do i=1, N
         a2(i) = i
      enddo
!$xmp loop on t4(i)
      do i=1, N
         b2(i) = dble(i)
      enddo
!$xmp loop on t6(i)
      do i=1, N
         c2(i) = real(i)
      enddo

!$xmp loop on t2(i)
      do i=1, N
!$xmp gmove out
         a1(i:i) = a2(i:i)
      enddo

!$xmp loop on t4(i)
      do i=1, N
!$xmp gmove out
         b1(i:i) = b2(i:i)
      enddo

!$xmp loop on t5(i)
      do i=1, N
!$xmp gmove in
         c1(i:i) = c2(i:i)
      enddo

      result = 'OK '
!$xmp loop on t1(i)
      do i=1, N
         if(a1(i) .ne. i) result = 'NG1'
      enddo
!$xmp loop on t3(i)
      do i=1, N
         if(abs(b1(i)-dble(i)) .gt. 0.00000001) result = 'NG2'
      enddo
!$xmp loop on t5(i)
      do i=1, N
         if(abs(c1(i)-real(i)) .gt. 0.001) result = 'NG3'
      enddo
      
      print *, xmp_node_num(), 'testp093.f ', result

      end
