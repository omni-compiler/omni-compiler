      program main
      include 'xmp_lib.h'
!$xmp nodes p(*)
      integer procs, id, mask1, val1, result
      logical l1

      if(xmp_num_nodes().gt.31) then
         print *, 'You have to run this program by less than 32 nodes.'
         call exit(1)
      endif

      procs = xmp_num_nodes()
      id = xmp_node_num()-1
      result = 0
      do i=0, 2**procs-1
         mask1 = lshift(1, id)
         val1 = iand(i, mask1)
         if(val1 .eq. 0) then
            l1 = .false.
         else
            l1 = .true.
         endif
!$xmp reduction(.and.: l1)
         if(i.eq.2**procs-1) then
            if(.not.l1) result = -1 ! NG
         else
            if(l1) result = -1 ! NG
         endif
      enddo

!$xmp reduction(+:result)
!$xmp task on p(1)
      if( result .eq. 0 ) then
         write(*,*) "PASS"
      else
         write(*,*) "ERROR"
         call exit(1)
      endif
!$xmp end task

      end
