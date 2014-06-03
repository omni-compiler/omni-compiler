      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer random_array(N), ans_val
!$xmp nodes p(*)
!$xmp template t1(N,N,N)
!$xmp template t2(N,N,N)
!$xmp template t3(N,N,N)
!$xmp distribute t1(block,*,*) onto p
!$xmp distribute t2(*,block,*) onto p
!$xmp distribute t3(*,*,block) onto p
      integer a(N), sa
      real*8  b(N), sb
      real*4  c(N), sc
      integer ia, ib, ic, ii
!$xmp align a(i) with t1(i,*,*)
!$xmp align b(i) with t2(*,i,*)
!$xmp align c(i) with t3(*,*,i)
      character(len=2) result

      result = 'OK'
      do k=114, 10000, 17
         random_array(1) = k
         do i=2, N
            random_array(i) = mod(random_array(i-1)**2, 100000000)
            random_array(i) = mod((random_array(i)-mod(random_array(i),100))/100, 10000)
         enddo

!$xmp loop (i) on t1(i,:,:)
         do i=1, N
            a(i) = random_array(i)
         enddo

!$xmp loop (i) on t2(:,i,:)
         do i=1, N
            b(i) = dble(random_array(i))
         enddo

!$xmp loop (i) on t3(:,:,i)
         do i=1, N
            c(i) = real(random_array(i))
         enddo

         ans_val = 2147483647
         ii = 1
         do i=1, N
            if(ans_val .gt. random_array(i)) then
               ii = i
               ans_val = random_array(i)
            endif
         enddo

         sa = 2147483647
         sb = 10000000000.0
         sc = 10000000000.0
         ia = 1
         ib = 1
         ic = 1

!$xmp loop (i) on t1(i,:,:) reduction(firstmin: sa /ia/)
         do i=1, N
            if(sa .gt. a(i)) then
               ia = i
               sa = a(i)
            endif
         enddo

!$xmp loop (i) on t2(:,i,:) reduction(firstmin: sb /ib/)
         do i=1, N
            if(sb .gt. b(i)) then
               ib = i
               sb = b(i)
            endif
         enddo

!$xmp loop (i) on t3(:,:,i) reduction(firstmin: sc /ic/)
         do i=1, N
            if(sc .gt. c(i)) then
               ic = i
               sc = c(i)
            endif
         enddo

         if(  sa .ne. ans_val .or. sb .ne. dble(ans_val) .or. sc .ne. real(ans_val) .or. ia .ne. ii .or. ib .ne. ii .or. ic .ne. ii) then
            result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp164.f ', result

      end
