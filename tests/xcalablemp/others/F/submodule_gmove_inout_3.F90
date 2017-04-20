#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
 module subm0_gmove_inout_3
  integer, parameter :: N = 4
! By GNU Fortran (GCC) 6.2.0, 
! "submodule_gmove_inout_3.F90:sub_subm0_gmove_inout_3", line 28: failed to
! import module 'subm0_gmove_inout_3'
! will strangely be issued without the interface declaration below.
! It seems to trigger *.smod file to be output.
  interface
    module function func1()
    end function
  end interface

!$xmp nodes p0(8)

!$xmp nodes p1(2,2) = p0(1:4)
!$xmp nodes p2(2,2) = p0(5:8)

!$xmp template t1(N,N,N)
!$xmp distribute t1(*,block,block) onto p1

  integer a(N,N,N)
!$xmp align a(i,j,k) with t1(i,j,k)
!$xmp shadow a(0,2:1,1:0)

!$xmp template t2(N,N,N)
!$xmp distribute t2(block,cyclic,*) onto p2

  integer b(N,N,N)
!$xmp align b(i,j,k) with t2(i,j,k)
!$xmp shadow b(0:1,0,0)
 end module subm0_gmove_inout_3

 submodule(subm0_gmove_inout_3) sub_subm0_gmove_inout_3

 contains


!--------------------------------------------------------
subroutine gmove_in

use subm0_gmove_inout_3
integer :: result = 0

!$xmp loop (i,j,k) on t1(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           a(i,j,k) = 777
        end do
     end do
  end do

!$xmp loop (i,j,k) on t2(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           b(i,j,k) = i*10000 + j *100 + k
        end do
     end do
  end do

!$xmp barrier

!$xmp task on p1

#ifdef _MPI3
!$xmp gmove in
  a = b
#endif

!$xmp end task

!$xmp barrier

!$xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           if (a(i,j,k) /= i*10000 + j*100 + k) then
              write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p0(1) nocomm
  if (result /= 0) then
     write(*,*) "ERROR in gmove_in"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_in


!--------------------------------------------------------
subroutine gmove_in_async

use subm0_gmove_inout_3
integer :: result = 0

!$xmp loop (i,j,k) on t1(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           a(i,j,k) = 777
        end do
     end do
  end do

!$xmp loop (i,j,k) on t2(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           b(i,j,k) = i*10000 + j *100 + k
        end do
     end do
  end do

!$xmp barrier

!$xmp task on p1

#ifdef _MPI3
!$xmp gmove in async(10)
  a = b
#endif

!$xmp wait_async(10)

!$xmp end task

!$xmp barrier

!$xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           if (a(i,j,k) /= i*10000 + j*100 + k) then
              !write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p0(1) nocomm
  if (result /= 0) then
     write(*,*) "ERROR in gmove_in"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_in_async


!--------------------------------------------------------
subroutine gmove_out

use subm0_gmove_inout_3
integer :: result = 0

!$xmp loop (i,j,k) on t1(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           a(i,j,k) = 777
        end do
     end do
  end do

!$xmp loop (i,j,k) on t2(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           b(i,j,k) = i*10000 + j *100 + k
        end do
     end do
  end do

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out
  a = b
#endif

!$xmp end task

!$xmp barrier

!$xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           if (a(i,j,k) /= i*10000 + j*100 + k) then
              !write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p0(1)
  if (result /= 0) then
     write(*,*) "ERROR in gmove_in"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_out


!--------------------------------------------------------
subroutine gmove_out_async

use subm0_gmove_inout_3
integer :: result = 0

!$xmp loop (i,j,k) on t1(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           a(i,j,k) = 777
        end do
     end do
  end do

!$xmp loop (i,j,k) on t2(i,j,k)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           b(i,j,k) = i*10000 + j *100 + k
        end do
     end do
  end do

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out async(10)
  a = b
#endif

!$xmp wait_async(10)

!$xmp end task

!$xmp barrier

!$xmp loop (i,j,k) on t1(i,j,k) reduction(+:result)
  do k = 1, N
     do j = 1, N
        do i = 1, N
           if (a(i,j,k) /= i*10000 + j*100 + k) then
              !write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p0(1)
  if (result /= 0) then
     write(*,*) "ERROR in gmove_out_async"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_out_async

 end submodule sub_subm0_gmove_inout_3

!--------------------------------------------------------
program test

use subm0_gmove_inout_3

#ifdef _MPI3
  call gmove_in
  call gmove_in_async
  call gmove_out
  call gmove_out_async

!$xmp task on p0(1)
  write(*,*) "PASS"
!$xmp end task
#else
  write(*,*) "Skipped"
#endif

end program test
#else
  write(*,*) "SKIPPED"
end
#endif

