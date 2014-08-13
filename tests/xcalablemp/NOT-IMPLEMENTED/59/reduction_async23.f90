! testp205.f
! xmp_desc_of()とxmp_gtol()のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(4)
!$xmp template t1(N)
!$xmp template t2(N)
!$xmp template t3(N)
c!$xmp template t4(N)
!$xmp distribute t1(block) onto p
!$xmp distribute t2(cyclic) onto p
!$xmp distribute t3(cyclic(3)) onto p
c!$xmp distribute t4(gblock((/100,200,300,400/))) onto p
      integer a(N)
      real*8  b(N)
      real*4  c(N)
c      complex*8 d(N)
!$xmp align a(i) with t1(i)
!$xmp align b(i) with t2(i)
!$xmp align c(i) with t3(i)
c!$xmp align d(i) with t4(i)
      integer(kind=xmp_desc_kind) da
      integer(kind=xmp_desc_kind) db
      integer(kind=xmp_desc_kind) dc
c      integer(kind=xmp_desc_kind) dd
      integer g_idx(1)
      integer l_idx(1)
      character(len=3) result

      da = xmp_desc_of(a)
      db = xmp_desc_of(b)
      dc = xmp_desc_of(c)
c      dd = xmp_desc_of(d)

      result = 'OK '
      do i=1, N
         g_idx(1) = i
         call xmp_gtol(da, g_idx, l_idx)
         if(mod(i, 250) .ne. l_idx(1)) result = 'NG1'

         call xmp_gtol(db, g_idx, l_idx)
         if(i/4 .ne. l_idx(1)) result = 'NG2'

         call xmp_gtol(dc, g_idx, l_idx)
         if((i/12)*3+mod(i,3) .ne. l_idx(1)) result = 'NG3'

c         call xmp_gtol(dd, g_idx, l_idx)
c         if(i.le.100) then
c            if(i .ne. l_idx(1)) result = 'NG4'
c         else if(i.gt.100 .and. i.le.300) then
c            if(i-100 .ne. l_idx(1)) result = 'NG5'
c         else if(i.gt.300 .and. i.le.600) then
c            if(i-300 .ne. l_idx(1)) result = 'NG6'
c         else
c            if(i-600 .ne. l_idx(1)) result = 'NG7'
c         endif
      enddo

      print *, xmp_all_node_num(), 'testp205.f ', result

      end
      
         
