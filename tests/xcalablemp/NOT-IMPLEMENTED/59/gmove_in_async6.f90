! testp103.f
! loop指示文とgmove指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t1(N,N)
!$xmp template t2(N,N)
!$xmp template t3(N,N)
!$xmp template t4(N,N)
!$xmp template t5(N,N)
!$xmp template t6(N,N)
!$xmp distribute t1(block,block) onto p
!$xmp distribute t2(cyclic,block) onto p
!$xmp distribute t3(block,cyclic) onto p
!$xmp distribute t4(cyclic,cyclic) onto p
!$xmp distribute t5(cyclic(5),block) onto p
!$xmp distribute t6(block,cyclic(7)) onto p
      integer a1(N,N), a2(N,N)
      real*8  b1(N,N), b2(N,N)
      real*4  c1(N,N), c2(N,N)
!$xmp align a1(i,j) with t1(i,j)
!$xmp align a2(i,j) with t2(i,j)
!$xmp align b1(i,j) with t3(i,j)
!$xmp align b2(i,j) with t4(i,j)
!$xmp align c1(i,j) with t5(i,j)
!$xmp align c2(i,j) with t6(i,j)
      character(len=3) result

!$xmp loop on t1(i,j)
      do j=1, N
         do i=1, N
            a1(i,j) = 0
         enddo
      enddo
!$xmp loop on t3(i,j)
      do j=1, N
         do i=1, N
            b1(i,j) = 0.0
         enddo
      enddo
!$xmp loop on t5(i,j)
      do j=1, N
         do i=1, N
            c1(i,j) = 0.0
         enddo
      enddo

!$xmp loop on t2(i,j)
      do j=1, N
         do i=1, N
            a2(i,j) = (j-1)*N+i
         enddo
      enddo
!$xmp loop on t4(i,j)
      do j=1, N
         do i=1, N
            b2(i,j) = dble((j-1)*N+i)
         enddo
      enddo
!$xmp loop on t6(i,j)
      do j=1, N
         do i=1, N
            c2(i,j) = real((j-1)*N+i)
         enddo
      enddo

!$xmp loop on t1(:,j)
      do j=1, N
!$xmp gmove in async(1)
         a1(1:N:2,j) = a2(1:N:2,j)
!$xmp gmove in async(2)
         a1(2:N:2,j) = a2(2:N:2,j)
!$xmp wait_async(1)
!$xmp wait_async(2)
      enddo

!$xmp loop on t3(:,j)
      do j=1, N
!$xmp gmove in async(1)
         b1(1:N:2,j) = b2(1:N:2,j)
!$xmp gmove in async(2)
         b1(2:N:2,j) = b2(2:N:2,j)
!$xmp wait_async(1)
!$xmp wait_async(2)
      enddo

!$xmp loop on t6(:,j)
      do j=1, N
!$xmp gmove out async(1)
         c1(1:N:2,j) = c2(1:N:2,j)
!$xmp gmove out async(2)
         c1(2:N:2,j) = c2(2:N:2,j)
!$xmp wait_async(1)
!$xmp wait_async(2)
      enddo

      result = 'OK '
!$xmp loop on t1(i,j)
      do j=1, N
         do i=1, N
            if(a1(i,j) .ne. (j-1)*N+i) result = 'NG1'
         enddo
      enddo
!$xmp loop on t3(i,j)
      do j=1, N
         do i=1, N
            if(abs(b1(i,j)-dble((j-1)*N+i)) .gt. 0.00000001)
     $           result = 'NG2'
         enddo
      enddo
!$xmp loop on t5(i,j)
      do j=1, N
         do i=1, N
            if(abs(c1(i,j)-real((j-1)*N+i)) .gt. 0.001) result = 'NG3'
         enddo
      enddo
      
      print *, xmp_node_num(), 'testp103.f ', result

      end
