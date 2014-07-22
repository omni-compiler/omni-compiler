! testp114.f
! reflect指示文、2次元分散

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4,*)
!$xmp template t(N,N)
!$xmp distribute t(block,block) onto p
      integer a(N,N)
      real*8  b(N,N)
      real*4  c(N,N)
!$xmp align a(i,j) onto t(i,j)
!$xmp align b(i,j) onto t(i,j)
!$xmp align c(i,j) onto t(i,j)
!$xmp shadow a(*,*)
!$xmp shadow b(1,1)
!$xmp shadow c(2,3)
      integer p2, w1, w2, pi, pj, procs
      character(len=3) result

      result = 'OK '

      procs = xmp_num_nodes()
      p2 = procs/4
      w1 = 250
      if(mod(N,p2).eq.0) then
         w2 = N/p2
      else
         w2 = N/p2+1
      endif
      pi = mod(xmp_node_num(), 4)
      pj = xmp_node_num()/4
      
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = 1
            b(i,j) = 1.0
            c(i,j) = 1.0
         enddo
      enddo

!$xmp reflect (a)
!$xmp reflect (b)
!$xmp reflect (c)

!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            do jj=j-4, j+4
               do ii=i-4, i+4
                  if(  ii.lt.1 .or. ii.gt.N .or.
     $                 jj.lt.1 .or. jj.gt.N) then
                     cycle
                  else
                     if(a(ii,jj) .ne. 1) result = 'NG1'
                  endif
               enddo
            enddo
         enddo
      enddo
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            do jj=j-1, j+1
               do ii=i-1, i+1
                  if(  ii.lt.1 .or. ii.gt.N .or.
     $                 jj.lt.1 .or. jj.gt.N) then
                     cycle
                  else
                     if(b(ii,jj) .ne. 1.0) result = 'NG2'
                  endif
               enddo
            enddo
         enddo
      enddo
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            do jj=j-3, j+3
               do ii=i-2, i+2
                  if(  ii.lt.1 .or. ii.gt.N .or.
     $                 jj.lt.1 .or. jj.gt.N) then
                     cycle
                  else
                     if(c(ii,jj) .ne. 1.0) result = 'NG3'
                  endif
               enddo
            enddo
         enddo
      enddo
      
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            a(i,j) = 2
            b(i,j) = 2.0
            c(i,j) = 2.0
         enddo
      enddo

!$xmp reflect (a) width(1,1)
!$xmp reflect (b) width(1,1)
!$xmp reflect (c) width(1,2)

!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            do jj=j-1, j+1
               do ii=i-1, i+1
                  if(  ii.lt.1 .or. ii.gt.N .or.
     $                 jj.lt.1 .or. jj.gt.N) then
                     cycle
                  else
                     if(a(ii,jj) .ne. 2) result = 'NG4'
                  endif
               enddo
            enddo
         enddo
      enddo
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            do jj=j-1, j+1
               do ii=i-1, i+1
                  if(  ii.lt.1 .or. ii.gt.N .or.
     $                 jj.lt.1 .or. jj.gt.N) then
                     cycle
                  else
                     if(b(ii,jj) .ne. 2.0) result = 'NG5'
                  endif
               enddo
            enddo
         enddo
      enddo
!$xmp loop (i,j) on t(i,j)
      do j=1, N
         do i=1, N
            do jj=j-2, j+2
               do ii=i-1, i+1
                  if(  ii.lt.1 .or. ii.gt.N .or.
     $                 jj.lt.1 .or. jj.gt.N) then
                     cycle
                  else
                     if(c(ii,jj) .ne. 2.0) result = 'NG6'
                  endif
               enddo
            enddo
            if(i.ne.1 .and. i.eq.pi*w1+1) then
               if(c(i-2,j) .ne. 1.0) result = 'NG7'
            endif
            if(i.ne.N .and. i.eq.(pi+1)*w1) then
               if(c(i+2,j) .ne. 1.0) result = 'NG8'
            endif
            if(j.ne.1 .and. j.eq.pj*w2+1) then
               if(c(i,j-2) .ne. 1.0) result = 'NG9'
            endif
            if(j.ne.N .and. j.eq.(pj+1)*w2) then
               if(c(i,j+2) .ne. 1.0) result = 'NG '
            endif
         enddo
      enddo

      print *, xmp_node_num(), 'testp114.f ', result

      end
      
