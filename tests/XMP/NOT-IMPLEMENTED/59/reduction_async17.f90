! testp059.f
! reduction指示文(min)のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer random_array(N*N), ans_val, val
      integer a(N,N), sa
      real*8  b(N,N), sb
      real*4  c(N,N), sc
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(block,block) onto p
      character(len=2) result

      result = 'OK'
      do k=114, 10000, 113
         random_array(1) = k
         do i=2, N*N
            random_array(i) =
     $           mod(random_array(i-1)**2, 100000000)
            random_array(i) =
     $           mod((random_array(i)-mod(random_array(i),100))/100,
     $           10000)
         enddo

!$xmp loop (i,j) on t(i,j)
         do j=1, N
            do i=1, N
               l = (j-1)*N+i
               a(i,j) = random_array(l)
               b(i,j) = dble(random_array(l))
               c(i,j) = real(random_array(l))
            enddo
         enddo
         
         sa = 2147483647
         sb = 1000000000000.0
         sc = 1000000000000.0
!$xmp loop (i,j) on t(i,j)
         do j=1, N
            do i=1, N
               sa = min(sa, a(i,j))
               sb = min(sa, b(i,j))
               sc = min(sa, c(i,j))
            enddo
         enddo
!$xmp reduction(min: sa, sb, sc) async(1)
         ans_val = 2147483647
         do j=1, N
            do i=1, N
               l = (j-1)*N+i
               ans_val = min(ans_val, random_array(l))
            enddo
         enddo
!$xmp wait_async(1)

         if(  sa .ne. ans_val .or.
     $        sb .ne. dble(ans_val) .or.
     $        sc .ne. real(ans_val) ) then
            result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp059.f ', result

      end
