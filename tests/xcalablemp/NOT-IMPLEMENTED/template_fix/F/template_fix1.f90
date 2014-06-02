      program main
      include 'xmp_lib.h'
!$xmp nodes p(*)
!$xmp template t(:)
!$xmp distribute t(block) onto p
      integer N, s
      integer,allocatable:: a(:)
!$xmp align a(i) with t(i)
      character(len=2) result

      N = 0
      do i=1, 1000
         N = N+1
      enddo
      
!$xmp template_fix (block) t(N)
      allocate(a(N))

!$xmp loop (i) on t(i)
      do i=1, N
         a(i) = i
      enddo
         
      s = 0
!$xmp loop (i) on t(i) reduction(+: s)
      do i=1, N
         s = s+a(i)
      enddo

      result = 'OK'
      if(s .ne. 500500) then
         result = 'NG'
      endif

      print *, xmp_all_node_num(), 'testp001.f ', result
      deallocate(a)

      end
