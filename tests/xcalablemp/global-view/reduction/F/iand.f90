      program main
      include 'xmp_lib.h'
!$xmp nodes p(4,*)
      integer procs, id, mask, val, result

      if(xmp_num_nodes().gt.31) then
         print *, 'You have to run this program by less than 32 nodes.'
         call exit(1)
      endif

      procs = xmp_num_nodes()
      id = xmp_node_num()-1
      result = 0
      do i=0, 2**procs-1
         mask = lshift(1, id)
         val = not(iand(i, mask))
!$xmp reduction(iand: val)
         if(not(val) .ne. i) then
            result = -1  ! NG
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
