      program main
      include 'xmp_lib.h'
!$xmp nodes p(*)
!$xmp template t(:,:)
!$xmp distribute t(*,block) onto p
      integer N, s
      integer,allocatable:: a(:,:)
!$xmp align a(i) with t(i)
      character(len=2) result

      N = 0
      do i=1, 1000
         N = N+1
      enddo
      
!$xmp template_fix (*,block) t(N,N)
      allocate(a(N,N))

!$xmp loop (j) on t(j)
      do j=1, N
         do i=1, N
            a(i,j) = i
         enddo
      enddo
         
      s = 0
!$xmp loop (j) on t(j) reduction(+: s)
      do j=1, N
         do i=1, N
            s = s+a(i,j)
         enddo
      enddo

      result = 'OK'
      if(s .ne. 500500000) then
         result = 'NG'
      endif

      print *, xmp_node_num(), 'testp002.f ', result
      deallocate(a)

      end
