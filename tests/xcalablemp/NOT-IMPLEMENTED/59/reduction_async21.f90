! testp101.f
! loop指示文とreflect指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t1(N,N)
!$xmp template t2(N,N)
!$xmp template t3(N,N)
!$xmp distribute t1(block,block) onto p
!$xmp distribute t2(block,block) onto p
!$xmp distribute t3(block,block) onto p
      integer a1(N,N), a2(N,N)
      real*8  b1(N,N), b2(N,N)
      real*4  c1(N,N), c2(N,N)
!$xmp align a1(i,j) with t1(i,j)
!$xmp align a2(i,j) with t1(i,j)
!$xmp align b1(i,j) with t2(i,j)
!$xmp align b2(i,j) with t2(i,j)
!$xmp align c1(i,j) with t3(i,j)
!$xmp align c2(i,j) with t3(i,j)
!$xmp shadow a2(1,1)
!$xmp shadow b2(2,2)
!$xmp shadow c2(3,3)
      character(len=3) result

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            a1(i,j) = 0
            a2(i,j) = (j-1)*N+i
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            b1(i,j) = 0.0
            b2(i,j) = dble((j-1)*N+i)
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            c1(i,j) = 0.0
            c2(i,j) = real((j-1)*N+i)
         enddo
      enddo

      result = 'OK '
!$xmp loop (i,j) on t1(i,j)
      do j=2, N-1
         do i=2, N-1
!$xmp reflect (a2) async(1)
            if(a1(i,j) .ne. 0) result = 'NG1'
!$xmp wait_async(1)
            do jj=j-1, j+1
               do ii=i-1, i+1
                  a1(i,j) = a1(i,j)+a2(ii,jj)
               enddo
            enddo
            a1(i,j) = a1(i,j)/9
         enddo
      enddo

!$xmp loop (i,j) on t2(i,j)
      do j=3, N-2
         do i=3, N-2
!$xmp reflect (b2) async(1)
            if(b1(i,j) .ne. 0) result = 'NG2'
!$xmp wait_async(1)
            do jj=j-2, j+2
               do ii=i-2, i+2
                  b1(i,j) = b1(i,j)+b2(ii,jj)
               enddo
            enddo
            b1(i,j) = b1(i,j)/25.0
         enddo
      enddo

!$xmp loop (i,j) on t3(i,j)
      do j=4, N-3
         do i=4, N-3
!$xmp reflect (c2) async(1)
            if(c1(i,j) .ne. 0) result = 'NG3'
!$xmp wait_async(1)
            do jj=j-3, j+3
               do ii=i-3, i+3
                  c1(i,j) = c1(i,j)+c2(ii,jj)
               enddo
            enddo
            c1(i,j) = c1(i,j)/49.0
         enddo
      enddo

!$xmp loop (i,j) on t1(i,j)
      do j=1, N
         do i=1, N
            if(i.eq.1.or.i.eq.N.or.j.eq.1.or.j.eq.N) then
               if(a1(i,j) .ne. 0) result = 'NG4'
            else
               if(a1(i,j) .ne. (j-1)*N+i) result = 'NG5'
            endif
         enddo
      enddo
!$xmp loop (i,j) on t2(i,j)
      do j=1, N
         do i=1, N
            if(i.le.2.or.i.ge.N-1.or.j.le.2.or.j.ge.N-1) then
               if(b1(i,j) .ne. 0.0) result = 'NG6'
            else
               if(abs(b1(i,j)-dble((j-1)*N+i)) .gt. 0.000001)
     $              result = 'NG7'
            endif
         enddo
      enddo
!$xmp loop (i,j) on t3(i,j)
      do j=1, N
         do i=1, N
            if(i.le.3.or.i.ge.N-2.or.j.le.3.or.j.ge.N-2) then
               if(c1(i,j) .ne. 0.0) result = 'NG8'
            else
               if(abs(c1(i,j)-real((j-1)*N+i)) .gt. 1) then
                  result = 'NG9'
               endif
            endif
         enddo
      enddo
      
      print *, xmp_node_num(), 'testp101.f ', result

      end
