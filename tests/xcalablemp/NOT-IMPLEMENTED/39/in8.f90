! testp099.f
! loop指示文とpost/wait指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N)
!$xmp distribute t(cyclic) onto p
      integer a(N), aa
      integer procs
!$xmp align a(i) with t(i)
      character(len=3) result

      procs = xmp_num_nodes()
      if(xmp_node_num().eq.1) then
         a(1) = 1
      endif
!$xmp barrier
!$xmp loop on t(i)
      do i=2, N
         if(i.ne.2) then
!$xmp wait(p(mod(i-1,procs)), 1)
         endif
!$xmp gmove in
         aa = a(i-1)
         a(i) = aa+1
         if(i.ne.N) then
!$xmp post(p(mod(i+1,procs)), 1)
         endif
      enddo

      result = 'OK '
!$xmp loop on t(i)
      do i=1, N
         if(a(i) .ne. i) result = 'NG '
      enddo

      print *, xmp_node_num(), 'testp099.f ', result

      end
      
