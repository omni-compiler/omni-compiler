program main
  include 'xmp_lib.h'
!$xmp nodes p(4,4)
!$xmp template t(:,:,:)
!$xmp distribute t(*,cyclic,block) onto p
  integer N, s, result
  integer,allocatable:: a(:,:)
!$xmp align a(i,j) with t(*,i,j)

  N = 100
      
!$xmp template_fix (*,cyclic,block) t(N,N,N)
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

  result = 0
  if(s .ne. 85000) then
     result = -1
  endif

!$xmp reduction(+:result)
!$xmp task on p(1,1)
  if( result .eq. 0 ) then
     write(*,*) "PASS"
  else
     write(*,*) "ERROR"
     call exit(1)
  endif
!$xmp end task
  deallocate(a)
  
end program main
