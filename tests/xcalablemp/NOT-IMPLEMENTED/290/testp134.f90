      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
      integer aa(N), a
      real*4 bb(N), b
      real*8 cc(N), c
      integer procs, id
      character(len=2) result

      id = xmp_node_num()
      procs = xmp_num_nodes()

      result = 'OK'
      j=2
      a = xmp_node_num()
      b = real(a)
      c = dble(a)
      do i=1, N
         aa(i) = a+i-1
         bb(i) = real(a+i-1)
         cc(i) = dble(a+i-1)
      enddo

!$xmp bcast (a) from p(j) on p(2:procs-1)
!$xmp bcast (b) from p(j) on p(2:procs-1)
!$xmp bcast (c) from p(j) on p(2:procs-1)
!$xmp bcast (aa) from p(j) on p(2:procs-1)
!$xmp bcast (bb) from p(j) on p(2:procs-1)
!$xmp bcast (cc) from p(j) on p(2:procs-1)

      if(id .ge. 2 .and. id .le. procs-1) then
         if(a .ne. j) result = 'NG'
         if(b .ne. real(j)) result = 'NG'
         if(c .ne. dble(j)) result = 'NG'
         do i=1, N
            if(aa(i) .ne. j+i-1) result = 'NG'
            if(bb(i) .ne. real(j+i-1)) result = 'NG'
            if(cc(i) .ne. dble(j+i-1)) result = 'NG'
         enddo
      else
         if(a .ne. xmp_node_num()) result = 'NG'
         if(b .ne. real(a)) result = 'NG'
         if(c .ne. dble(a)) result = 'NG'
         do i=1, N
            if(aa(i) .ne. a+i-1) result = 'NG'
            if(bb(i) .ne. real(a+i-1)) result = 'NG'
            if(cc(i) .ne. dble(a+i-1)) result = 'NG'
         enddo
      endif

      print *, xmp_node_num(), 'testp134.f ', result

      end
