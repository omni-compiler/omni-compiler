! testp081.f
! task指示文とgmove指示文の組合せテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,4)
!$xmp template t1(N,N)
!$xmp distribute t1(block,gblock((/200,700,50,50/))) onto p
!$xmp template t2(N,N)
!$xmp distribute t2(block,gblock((/700,200,50,50/))) onto p
!$xmp template t3(N,N)
!$xmp distribute t3(gblock((/250,250,250,250/)),gblock((/50,700,200,50/))) onto p
      integer a1(N,N), a2(N,N)
      real*8  b1(N,N), b2(N,N)
      real*4  c1(N,N), c2(N,N)
!$xmp align a1(i,j) with t1(i,j)
!$xmp align a2(i,j) with t2(i,j)
!$xmp align b1(i,j) with t3(i,j)
!$xmp align b2(i,j) with t1(i,j)
!$xmp align c1(i,j) with t2(i,j)
!$xmp align c2(i,j) with t3(i,j)
      character(len=2) result

      if(xmp_num_nodes().ne.16) then
         print *, 'You have to run this program by 16 nodes.'
      endif
      
!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            a1(i,j) = xmp_node_num()
            b2(i,j) = -1.0
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            a2(i,j) = -1
            c1(i,j) = real(xmp_node_num())
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            b1(i,j) = dble(xmp_node_num())
            c2(i,j) = -1.0
         enddo
      enddo

!$xmp task on p(1,1)
!$xmp gmove in async(2)
      b2(1:250,1:200) = b1(501:750,751:950)  ! p(3,3)
!$xmp wait_async(2)
!$xmp end task

!$xmp task on p(2,2)
!$xmp gmove in async(1)
      a2(251:500,701:900) = a1(751:1000,1:200) ! p(4,1)
!$xmp wait_async(1)
!$xmp end task

!$xmp task on p(3,3)
!$xmp gmove in async(3)
      c2(501:750,751:950) = c1(251:500,701:900) ! p(2,2)
!$xmp wait_async(3)
!$xmp end task

      result = 'OK'
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            if(i.ge.251.and.i.le.500.and.j.ge.701.and.j.le.900) then
               if(a2(i,j) .ne. 13) then
                  result = 'NG'
               endif
            else
               if(a2(i,j) .ne. -1) then
                  result = 'NG'
               endif
            endif
         enddo
      enddo
         
!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            if(i.ge.1.and.i.le.200.and.j.ge.1.and.j.le.250) then
               if(b2(i,j) .ne. 4.0) then
                  result = 'NG'
               endif
            else
               if(b2(i,j) .ne. -1.0) then
                  result = 'NG'
               endif
            endif
         enddo
      enddo

!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            if(i.ge.501.and.i.le.750.and.j.ge.751.and.j.le.950) then
               if(c2(i,j) .ne. 6.0) then
                  result = 'NG'
               endif
            else
               if(c2(i,j) .ne. -1.0) then
                  result = 'NG'
               endif
            endif
         enddo
      enddo

      print *, xmp_node_num(), 'testp081.f ', result

      end
