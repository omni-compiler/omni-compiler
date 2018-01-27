! testp111.f
! reflect指示文、1次元分散、width、asyncあり

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(block) onto p
      integer a(N)
      real*8  b(N)
      real*4  c(N)
!$xmp align a(i) with t(i)
!$xmp align b(i) with t(i)
!$xmp align c(i) with t(i)
!$xmp shadow a(1)
!$xmp shadow b(2)
!$xmp shadow c(3)
      integer w
      character(len=3) result

      if(mod(N,xmp_num_nodes()) .eq. 0) then
         w = N/xmp_num_nodes()
      else
         w = N/xmp_num_nodes()+1
      endif
      
      result = 'OK '
      k = 1
!$xmp loop on t(i)
      do i=1, N
         a(i) = k
         b(i) = dble(k)
         c(i) = real(k)
      enddo
      
!$xmp reflect (a) async(1)
!$xmp reflect (b) async(2)
!$xmp reflect (c) async(3)

!$xmp wait_async(1)
!$xmp loop on t(i)
      do i=2, N-1
         do ii=i-1,i+1
            if(a(ii) .ne. k) result = 'NG1'
         enddo
      enddo
!$xmp wait_async(2)
!$xmp loop on t(i)
      do i=3, N-2
         do ii=i-2,i+2
            if(b(ii) .ne. dble(k)) result = 'NG2'
         enddo
      enddo
!$xmp wait_async(3)
!$xmp loop on t(i)
      do i=4, N-3
         do ii=i-3,i+3
            if(c(ii) .ne. real(k)) result = 'NG3'
         enddo
      enddo
      
      k = 2
!$xmp loop on t(i)
      do i=1, N
         a(i) = k
         b(i) = dble(k)
         c(i) = real(k)
      enddo
      
!$xmp reflect (a) async(1)
!$xmp reflect (b) width(2) async(2)
!$xmp reflect (c) width(2) async(3)
!$xmp wait_async(1)

!$xmp loop on t(i)
      do i=2, N-1
         do ii=i-1,i+1
            if(a(ii) .ne. k) result = 'NG4'
         enddo
      enddo
!$xmp wait_async(2)
!$xmp loop on t(i)
      do i=3, N-2
         do ii=i-2,i+2
            if(b(ii) .ne. dble(k)) result = 'NG5'
         enddo
      enddo
!$xmp wait_async(3)
!$xmp loop on t(i)
      do i=4, N-3
         do ii=i-2,i+2
            if(c(ii) .ne. real(k)) result = 'NG6'
         enddo
         if(i.eq.4 .or. i.eq.w+1) then
            if(c(i-3) .ne. 1.0) result = 'NG7'
         endif
         if(i.eq.N-3 .or. i.eq.w*xmp_node_num()) then
            if(c(i+3) .ne. 1.0) result = 'NG8'
         endif
      enddo

      print *, xmp_node_num(), 'testp111.f ', result

      end
