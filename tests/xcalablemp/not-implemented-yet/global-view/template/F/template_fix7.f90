      program main
      include 'xmp_lib.h'
!$xmp nodes p(4,4)
!$xmp template t(:,:,:)
!$xmp distribute t(*,cyclic(3),cyclic(7)) onto p
      integer N, s
      integer,allocatable:: a(:,:)
!$xmp align a(i,j) with t(*,i,j)
      character(len=2) result

      if(xmp_all_num_nodes().ne.16) then
         print *, 'You have to run this program by 16 nodes.'
      endif

      N = 0
      do i=1, 1000
         N = N+1
      enddo
      
!$xmp template_fix (*,cyclic(3), cyclic(7)) t(N,N,N)
      allocate(a(N,N))

!$xmp loop (i,j) on t(:,i,j)
      do j=1, N
         do i=1, N
            a(i,j) = xmp_node_num()
         enddo
      enddo
         
      s = 0
!$xmp loop (i,j) on t(:,i,j) reduction(+: s)
      do j=1, N
         do i=1, N
            s = s+a(i,j)
         enddo
      enddo

      result = 'OK'
      if(s .ne. 45000000) then
         result = 'NG'
      endif

      print *, xmp_all_node_num(), 'testp007.f ', result
      deallocate(a)

      end
