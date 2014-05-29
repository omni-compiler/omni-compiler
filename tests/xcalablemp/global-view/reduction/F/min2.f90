      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
      integer random_array(N*N), ans_val, val, result
      integer a(N,N), sa
      real*8  b(N,N), sb
      real*4  c(N,N), sc
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(block,block) onto p

      result = 0
      do k=114, 10000, 113
         random_array(1) = k
         do i=2, N*N
            random_array(i) = mod(random_array(i-1)**2, 100000000)
            random_array(i) = mod((random_array(i)-mod(random_array(i),100))/100, 10000)
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
         
         ans_val = 2147483647
         do j=1, N
            do i=1, N
               l = (j-1)*N+i
               ans_val = min(ans_val, random_array(l))
            enddo
         enddo

         sa = 2147483647
         sb = 10000000000.0
         sc = 10000000000.0
!$xmp loop (i,j) on t(i,j)
         do j=1, N
            do i=1, N
               sa = min(sa, a(i,j))
               sb = min(sb, b(i,j))
               sc = min(sc, c(i,j))
            enddo
         enddo
!$xmp reduction(min: sa, sb, sc)

         if(  sa .ne. ans_val .or. sb .ne. dble(ans_val) .or. sc .ne. real(ans_val) ) then
            result = -1 ! NG
         endif
      enddo

!$xmp reduction(+:result)
!$xmp task on p(1,1)
      if( result .eq. 0 ) then
         write(*,*) "PASS"
      else
         write(*,*) "ERROR"
         call exit(1)
      endif
!$xmp end task

      end
