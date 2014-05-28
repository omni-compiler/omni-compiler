! testp070.f
! task指示文とgmove指示文の組合せテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4)
!$xmp template t1(N,N,N)
!$xmp distribute t1(*,*,gblock((/200,700,50,50/))) onto p
!$xmp template t2(N,N,N)
!$xmp distribute t2(*,*,gblock((/700,200,50,50/))) onto p
!$xmp template t3(N,N,N)
!$xmp distribute t3(*,*,gblock((/50,700,200,50/))) onto p
      integer a1(N), a2(N)
      real*8  b1(N), b2(N)
      real*4  c1(N), c2(N)
!$xmp align a1(i) with t1(*,*,i)
!$xmp align a2(i) with t2(*,*,i)
!$xmp align b1(i) with t3(*,*,i)
!$xmp align b2(i) with t1(*,*,i)
!$xmp align c1(i) with t2(*,*,i)
!$xmp align c2(i) with t3(*,*,i)
      character(len=2) result

      if(xmp_num_nodes().ne.4) then
         print *, 'You have to run this program by 4 nodes.'
      endif
      
!$xmp loop on t1(:,:,i)
      do i=1, N
         a1(i) = xmp_node_num()
         b2(i) = -1.0
      enddo
!$xmp loop on t2(:,:,i)
      do i=1, N
         a2(i) = -1
         c1(i) = real(xmp_node_num())
      enddo
!$xmp loop on t3(:,:,i)
      do i=1, N
         b1(i) = dble(xmp_node_num())
         c2(i) = -1.0
      enddo

!$xmp task on p(1)
!$xmp gmove in
      b2(1:200) = b1(751:950)
!$xmp end task

!$xmp task on p(2)
!$xmp gmove in
      a2(701:900) = a1(1:200)
!$xmp end task

!$xmp task on p(3)
!$xmp gmove in
      c2(751:950) = c1(701:900)
!$xmp end task

!$xmp barrier

      result = 'OK'
!$xmp loop on t2(:,:,i)
      do i=1, N
         if(i.ge.701 .and. i.le.900) then
            if(a2(i) .ne. 1) then
               result = 'NG'
            endif
         else
            if(a2(i) .ne. -1) then
               result = 'NG'
            endif
         endif
      enddo

!$xmp loop on t1(:,:,i)
      do i=1, N
         if(i.ge.1 .and. i.le.200) then
            if(b2(i) .ne. 3.0) then
               result = 'NG'
            endif
         else
            if(b2(i) .ne. -1.0) then
               result = 'NG'
            endif
         endif
      enddo

!$xmp loop on t3(:,:,i)
      do i=1, N
         if(i.ge.751 .and. i.le.950) then
            if(c2(i) .ne. 2.0) then
               result = 'NG'
            endif
         else
            if(c2(i) .ne. -1.0) then
               result = 'NG'
            endif
         endif
      enddo

      print *, xmp_node_num(), 'testp070.f ', result

      end
