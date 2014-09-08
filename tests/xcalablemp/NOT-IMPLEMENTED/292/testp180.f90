      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer random_array(N**2), ans_val
!$xmp nodes p(4,4,*)
!$xmp template t1(N,N,N)
!$xmp template t2(N,N,N)
!$xmp template t3(N,N,N)
!$xmp distribute t1(cyclic,cyclic,cyclic) onto p
!$xmp distribute t2(block,cyclic,cyclic) onto p
!$xmp distribute t3(cylic,cyclic,block) onto p
      integer a(N,N), sa
      real*8  b(N,N), sb
      real*4  c(N,N), sc
      integer ia, ib, ic, ii
      integer ja, jb, jc, jj
!$xmp align a(i,j) with t1(i,j,*)
!$xmp align b(i,j) with t2(*,i,j)
!$xmp align c(i,j) with t3(i,*,j)
      character(len=2) result

      result = 'OK'
      do k=114, 10000, 17
         random_array(1) = k
         do i=2, N**2
            random_array(i) = mod(random_array(i-1)**2, 100000000)
            random_array(i) = mod((random_array(i)-mod(random_array(i),100))/100, 10000)
         enddo

!$xmp loop (i,j) on t1(i,j,:)
         do j=1, N
            do i=1, N
               m = (j-1)*N+i
               a(i,j) = random_array(m)
            enddo
         enddo

!$xmp loop (i,j) on t2(:,i,j)
         do j=1, N
            do i=1, N
               m = (j-1)*N+i
               b(i,j) = dble(random_array(m))
            enddo
         enddo

!$xmp loop (i,j) on t3(i,:,j)
         do j=1, N
            do i=1, N
               m = (j-1)*N+i
               c(i,j) = real(random_array(m))
            enddo
         enddo

         ans_val = 2147483647
         ii = 1
         jj = 1
         do j=1, N
            do i=1, N
               m = (j-1)*N+i
               if(ans_val .ge. random_array(m)) then
                  ii = i
                  jj = j
                  ans_val = random_array(m)
               endif
            enddo
         enddo

         sa = 2147483647
         sb = 10000000000.0
         sc = 10000000000.0
         ia = 1
         ib = 1
         ic = 1
         ja = 1
         jb = 1
         jc = 1
!$xmp loop (i,j) on t1(i,j,:) reduction(lastmin: sa /ia, ja/)
         do j=1, N
            do i=1, N
               if(sa .ge. a(i,j)) then
                  ia = i
                  ja = j
                  sa = a(i,j)
               endif
            enddo
         enddo
!$xmp loop (i,j) on t2(:,i,j) reduction(lastmin: sb /ib, jb/)
         do j=1, N
            do i=1, N
               if(sb .ge. b(i,j)) then
                  ib = i
                  jb = j
                  sb = b(i,j)
               endif
            enddo
         enddo
!$xmp loop (i,j) on t3(i,:,j) reduction(lastmin: sc /ic, jc/)
         do j=1, N
            do i=1, N
               if(sc .ge. c(i,j)) then
                  ic = i
                  jc = j
                  sc = c(i,j)
               endif
            enddo
         enddo

         if(  sa .ne. ans_val .or. sb .ne. dble(ans_val) .or. sc .ne. real(ans_val) .or. ia .ne. ii .or. ib .ne. ii .or. ic .ne. ii .or. ja .ne. jj .or. jb .ne. jj .or. jc .ne. jj) then
            result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp180.f ', result

      end
