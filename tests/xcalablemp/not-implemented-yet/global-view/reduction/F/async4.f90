! testp033.f
! reduction指示文(iand)のテスト

      program main
      include 'xmp_lib.h'
!$xmp nodes p(*)
      integer procs, id
      integer mask1, val1
      integer mask2, val2
      character(len=2) result

      if(xmp_num_nodes().gt.31) then
         print *, 'You have to run this program by less than 32 nodes.'
      endif


      procs = xmp_num_nodes()
      id = xmp_node_num()-1
      result = 'OK'
      do i=0, 2**procs-1, 2
         mask1 = lshift(1, id)
         val1 = not(iand(i, mask1))
!$xmp reduction(iand: val1) async(1)
         mask2 = lshift(1, id)
         val2 = not(iand(i+1, mask2))
!$xmp reduction(iand: val2) async(2)
!$xmp wait_async(1)
         if(not(val1) .ne. i) then
            result = 'NG'
         endif
!$xmp wait_async(2)
         if(not(val2) .ne. i+1) then
            result = 'NG'
         endif
      enddo

      print *, xmp_node_num(), 'testp033.f ', result

      end
