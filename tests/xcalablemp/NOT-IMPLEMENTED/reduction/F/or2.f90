! testp064.f
! reduction指示文(.or.)のテスト

      program main
      include 'xmp_lib.h'
!$xmp nodes p(4,*)
      integer procs, id
      integer mask1, val1
      logical l1
      character(len=2) result

      if(xmp_num_nodes().gt.31) then
         print *, 'You have to run this program by less than 32 nodes.'
      endif


      procs = xmp_num_nodes()
      id = xmp_node_num()-1
      result = 'OK'
      do i=0, 2**procs-1
         mask1 = lshift(1, id)
         val1 = iand(i, mask1)
         if(val1 .eq. 0) then
            l1 = .false.
         else
            l1 = .true.
         endif
!$xmp reduction(.or.: l1)
         if(i.eq.0) then
            if(l1) result = 'NG'
         else
            if(.not.l1) result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp064.f ', result

      end
