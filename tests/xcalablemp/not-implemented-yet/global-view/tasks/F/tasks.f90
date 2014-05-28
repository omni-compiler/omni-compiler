! testp065.f
! tasks指示文およびtask指示文のテスト

      program main
      include 'xmp_lib.h'
      integer,parameter:: N=1000
!$xmp nodes p(*)
!$xmp template t(N,N,N)
!$xmp distribute t(*,*,block) onto p
      integer a(N), s, lb, ub, procs, w, ans
      character(len=2) result
!$xmp align a(i) with t(*,*,i)
!$xmp shadow a(1)

      s = 0
!$xmp loop on t(:,:,i)
      do i=1, N
         a(i) = i
      enddo

      procs = xmp_num_nodes()

      if(mod(N,procs) .eq. 0) then
         w = N/procs
      else
         w = N/procs+1
      endif

      lb = (xmp_node_num()-1)*w+1
      ub = xmp_node_num()*w
      
!$xmp tasks
!$xmp task on p(1)
      do i=lb, ub
         s = s+a(i)
      enddo
!$xmp end task
!$xmp task on p(2)
      if(procs .gt. 1) then
         do i=lb, ub
            s = s+a(i)
         enddo
      endif
!$xmp end task
!$xmp task on p(4)
      if(procs .gt. 3) then
         do i=lb, ub
            s = s+a(i)
         enddo
      endif
!$xmp end task
!$xmp end tasks

!$xmp reduction(+:s)

      if(procs .lt. 2) then
         ans = w*(w+1)/2
      else if(procs .lt.4) then
         ans = w*(w*2+1)
      else
         ans = w*(w*2+1) + w*2*(w*4+1) - w*3*(w*3+1)/2
      endif
      
      if(ans .eq. s) then
         result = 'OK'
      else
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp065.f ', result

      end
