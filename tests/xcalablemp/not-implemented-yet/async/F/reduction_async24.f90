! testp206.f
! 組込み手続のテスト

      program main
      include 'xmp_lib.h'
      real*8 time, tick, max_tick
      logical async_val
      character(len=2) result

      result = 'OK'
      time = xmp_wtime()
      tick = xmp_wtick()
      max_tick = tick
      
!$xmp reduction(max: tick) async(1)

 10   continue
      async_val = xmp_test_async(1)
      if(.not.async_val) goto 10

!$xmp wait_async(1)

      if(tick .ne. max_tick) result = 'NG'

      print *, xmp_node_num(), 'testp206.f ', result

      end
      
      
