      program main
      interface
         subroutine sub(x, N)
!$xmp nodes p(*)
!$xmp template t(:)
!$xmp distribute t(cyclic(3)) onto p
         integer N
         integer,allocatable:: x(:)
!$xmp align x(i) with t(i)
         end subroutine
      end interface
        
      include 'xmp_lib.h'
!$xmp nodes p(*)
!$xmp template t(:)
!$xmp distribute t(cyclic(3)) onto p
      integer N, s
      integer,allocatable:: a(:)
!$xmp align a(i) with t(i)
      character(len=2) result

      N = 0
      do i=1, 1000
         N = N+1
      enddo
      
!$xmp template_fix (cyclic(3)) t(N)
      allocate(a(N))

      call sub(a,N)
         
      s = 0
!$xmp loop (i) on t(i) reduction(+: s)
      do i=1, N
         s = s+a(i)
      enddo

      result = 'OK'
      if(s .ne. 500500) then
         result = 'NG'
         print *, s
      endif

      print *, xmp_all_node_num(), 'testp003.f ', result
      deallocate(a)

      end

      subroutine sub(x, N)
!$xmp nodes p(*)
!$xmp template t(:)
!$xmp distribute t(cyclic(3)) onto p
      integer N
      integer,allocatable:: x(:)
!$xmp align x(i) with t(i)

!$xmp loop (i) on t(i)
      do i=1, N
         x(i) = i
      enddo

      end
