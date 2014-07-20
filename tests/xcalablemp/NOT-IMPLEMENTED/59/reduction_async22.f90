! testp106.f
! loop指示文とreduction節(*)のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t1(block,block) onto p
!$xmp distribute t2(cyclic,block) onto p
!$xmp distribute t3(cyclic,cyclic(5)) onto p
      integer a(N,N), sa1, sa2
      real*8  b(N,N), sb1, sb2
      real*4  c(N,N), sc1, sc2
      character(len=2) result
!$xmp align a(i,j) with t1(i,j)
!$xmp align b(i,j) with t2(i,j)
!$xmp align c(i,j) with t3(i,j)

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            if(i.eq.j .and. mod(i,100).eq.0) then
               a(i,j) = 2
            else
               a(i,j) = 1
            endif
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            if(mod(j,2).eq.0) then
               if(mod(i,2).eq.0) then
                  b(i,j) = 2.0
               else
                  b(i,j) = 1.0
               endif
            else
               if(mod(i,2).eq.1) then
                  b(i,j) = 0.5
               else
                  b(i,j) = 1.0
               endif
            endif
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            if(mod(j,2).eq.0) then
               if(mod(i,4).eq.0) then
                  c(i,j) = 1.0
               else if(mod(i,4).eq.1) then
                  c(i,j) = 4.0
               else if(mod(i,4).eq.2) then
                  c(i,j) = 1.0
               else
                  c(i,j) = 0.25
               endif
            else
               if(mod(i,4).eq.0) then
                  c(i,j) = 0.25
               else if(mod(i,4).eq.1) then
                  c(i,j) = 1.0
               else if(mod(i,4).eq.2) then
                  c(i,j) = 4.0
               else
                  c(i,j) = 1.0
               endif
            endif
         enddo
      enddo

      sa1 = 1
      sb1 = 1.0
      sc1 = 1.0

!$xmp loop (j) on t1(:,j)
      do j=1, N
         sa2 = 1
!$xmp loop (i) on t1(i,j)
         do i=1, N
            sa2 = sa2*a(i,j)
         enddo
!$xmp reduction(*:sa2) async(1)
!$xmp loop (i) on t1(i,j)
         do i=1, N
            a(i,j) = 0
         enddo
!$xmp wait_async(1)
         sa1 = sa1*sa2
      enddo
      
!$xmp loop (j) on t2(:,j)
      do j=1, N
         sb2 = 1.0
!$xmp loop (i) on t2(i,j)
         do i=1, N
            sb2 = sb2*b(i,j)
         enddo
!$xmp reduction(*:sb2) async(2)
!$xmp loop (i) on t2(i,j)
         do i=1, N
            b(i,j) = 0.0
         enddo
!$xmp wait_async(2)
         sb1 = sb1*sb2
      enddo

!$xmp loop (j) on t3(:,j)
      do j=1, N
         sc2 = 1.0
!$xmp loop (i) on t3(i,j)
         do i=1, N
            sc2 = sc2*c(i,j)
         enddo
!$xmp reduction(*:sc2) async(3)
!$xmp loop (i) on t3(i,j)
         do i=1, N
            c(i,j) = 0.0
         enddo
!$xmp wait_async(3)
         sc1 = sc1*sc2
      enddo

      i = mod(xmp_node_num(), 4)
!$xmp reduction(*:sa1, sb1, sc1) on p(i,:)

      result = 'OK'
      if(  sa1 .ne. 1024 .or.
     $     abs(sb1-1.0) .gt. 0.000001 .or.
     $     abs(sc1-1.0) .gt. 0.0001) then
         print *, sa1, sb1, sc1
         result = 'NG'
      endif

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            if(a(i,j) .ne. 0) result = 'NG'
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            if(b(i,j) .ne. 0.0) result = 'NG'
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            if(c(i,j) .ne. 0.0) result = 'NG'
         enddo
      enddo

      print *, xmp_node_num(), 'testp106.f ', result

      end
