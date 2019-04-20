module mod0_gmove_test_out

  integer, parameter :: N = 64

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

  integer x(N,N,N)

  integer s

contains

  subroutine init_a
!$xmp loop on t1(i,j,k)
    do k = 1, N
       do j = 1, N
          do i = 1, N
             a(i,j,k) = 777
          end do
       end do
    end do
  end subroutine init_a

  subroutine init_b
!$xmp loop on t2(i,j,k)
    do k = 1, N
       do j = 1, N
          do i = 1, N
             b(i,j,k) = i*10000 + j *100 + k
          end do
       end do
    end do
  end subroutine init_b

  subroutine init_x
    do k = 1, N
       do j = 1, N
          do i = 1, N
             x(i,j,k) = i*10000 + j *100 + k
          end do
       end do
    end do
  end subroutine init_x

  subroutine init_x0
    x = 0
  end subroutine init_x

end module mod0_gmove_test_out


!--------------------------------------------------------
program gmove_test_out

  use mod0_gmove_test_out

#ifdef _MPI3
  call gmove_gs_gs
  call gmove_gs_ge
  call gmove_ge_ge

  call gmove_gs_ls
  call gmove_gs_le
  call gmove_ge_le

  call gmove_gs_s
  call gmove_ge_s

!$xmp task on p0(1) nocomm
  write(*,*) "PASS"
!$xmp end task
#else
!$xmp task on p0(1) nocomm
  write(*,*) "Skipped"
!$xmp end task
#endif

end program gmove_test_out


!--------------------------------------------------------
! global section = global section
!--------------------------------------------------------
subroutine gmove_gs_gs

  integer :: result = 0

  use mod0_gmove_test_out

  call init_a
  call init_b

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out
  a(1:N/4, N/2+1:N, 5:N-1) = b(N/2+1:N:2, 1:N/2, 1:N-5)
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp loop on t1(i,j,k) reduction (+:result)
  do k = 5, N-1
     do j = N/2+1, N
        do i = 1, N/4
           if (a(i,j,k) /= (N/2+1+(i-1)*2)*10000 + (j-N/2)*100 + (k-4)) then
!              write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k), (N/2+1+(i-1)*2)*10000 + (j-N/2)*100 + (k-4)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p1(1,1) nocomm
  if (result /= 0) then
     write(*,*) "ERROR in gmove_gs_gs"
     call exit(1)
  endif
!$xmp end task

!$xmp end task

end subroutine gmove_gs_gs


!--------------------------------------------------------
! global section = global element
!--------------------------------------------------------
subroutine gmove_gs_ge

  integer :: result = 0

  use mod0_gmove_test_out

  call init_a
  call init_b

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out
  a(1:N/4, N/2+1:N, 5:N-1) = b(3,4,5)
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp loop on t1(i,j,k) reduction(+:result)
  do k = 5, N-1
     do j = N/2+1, N
        do i = 1, N/4
           if (a(i,j,k) /= 3*10000 + 4*100 + 5) then
              write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p1(1,1) nocomm
  if (result /= 0) then
     write(*,*) "ERROR in gmove_gs_ge"
     call exit(1)
  endif
!$xmp end task

!$xmp end task

end subroutine gmove_gs_ge

!--------------------------------------------------------
! global element = global element
!--------------------------------------------------------
subroutine gmove_ge_ge

  use mod0_gmove_test_out

  call init_a
  call init_b

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out
  a(7,8,9) = b(3,4,5)
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp task on t1(7,8,9) nocomm
  if (a(7,8,9) /= 3*10000 + 4*100 + 5) then
!     write(*,*) "(", xmp_node_num(), ")", 7, 8, 9, a(7,8,9)
     write(*,*) "ERROR in gmove_ge_ge"
     call exit(1)
  end if
!$xmp end task

!$xmp end task

end subroutine gmove_ge_ge


!--------------------------------------------------------
! global section = local section
!--------------------------------------------------------
subroutine gmove_gs_ls

  integer :: result = 0

  use mod0_gmove_test_out

  call init_a
  call init_x

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out
  a(1:N/4, N/2+1:N, 5:N-1) = x(N/2+1:N:2, 1:N/2, 1:N-5)
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp loop on t1(i,j,k) reduction(+:result)
  do k = 5, N-1
     do j = N/2+1, N
        do i = 1, N/4
           if (a(i,j,k) /= (N/2+1+(i-1)*2)*10000 + (j-N/2)*100 + (k-4)) then
!              write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p1(1,1) nocomm
  if (result /= 0) then
     write(*,*) "ERROR in gmove_gs_ls"
     call exit(1)
  endif
!$xmp end task

!$xmp end task

end subroutine gmove_gs_ls


!--------------------------------------------------------
! global section = local element
!--------------------------------------------------------
subroutine gmove_gs_le

integer :: result = 0

  use mod0_gmove_test_out

  call init_a
  call init_x

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out
  a(1:N/4, N/2+1:N, 5:N-1) = x(3,4,5)
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp loop on t1(i,j,k) reduction(+:result)
  do k = 5, N-1
     do j = N/2+1, N
        do i = 1, N/4
           if (a(i,j,k) /= 3*10000 + 4*100 + 5) then
!              write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p1(1,1) nocomm
  if (result /= 0) then
     write(*,*) "ERROR in gmove_gs_le"
     call exit(1)
  endif
!$xmp end task

!$xmp end task

end subroutine gmove_gs_le

!--------------------------------------------------------
! global element = local element
!--------------------------------------------------------
subroutine gmove_ge_le

  use mod0_gmove_test_out

  call init_a
  call init_x

!$xmp barrier

!$xmp task on p2

#ifdef _MPI3
!$xmp gmove out
  a(7,8,9) = x(3,4,5)
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp task on t1(7,8,9) nocomm
  if (a(7,8,9) /= 3*10000 + 4*100 + 5) then
!     write(*,*) "(", xmp_node_num(), ")", 7, 8, 9, a(7,8,9)
     write(*,*) "ERROR in gmove_ge_le"
     call exit(1)
  end if
!$xmp end task

!$xmp end task

end subroutine gmove_ge_le


!--------------------------------------------------------
! global section = scalar
!--------------------------------------------------------
subroutine gmove_gs_s

integer :: result = 0

  use mod0_gmove_test_out

  call init_a

!$xmp barrier

!$xmp task on p2

  s = 111

#ifdef _MPI3
!$xmp gmove out
  a(1:N/4, N/2+1:N, 5:N-1) = s
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp loop on t1(i,j,k) reduction(+:result)
  do k = 5, N-1
     do j = N/2+1, N
        do i = 1, N/4
           if (a(i,j,k) /= 111) then
!              write(*,*) "(", xmp_node_num(), ")", i, j, k, a(i,j,k)
              result = 1
           end if
        end do
     end do
  end do

!$xmp task on p1(1,1) nocomm
  if (result /= 0) then
     write(*,*) "ERROR in gmove_gs_s"
     call exit(1)
  endif
!$xmp end task

!$xmp end task

end subroutine gmove_gs_s

!--------------------------------------------------------
! global element = scalar
!--------------------------------------------------------
subroutine gmove_ge_s

  use mod0_gmove_test_out

  call init_a

!$xmp barrier

!$xmp task on p2

  s = 111

#ifdef _MPI3
!$xmp gmove out
  a(7,8,9) = s
#endif

!$xmp end task

!$xmp barrier

!$xmp task on p1

!$xmp task on t1(7,8,9) nocomm
  if (a(7,8,9) /= 111) then
!     write(*,*) "(", xmp_node_num(), ")", 7, 8, 9, a(7,8,9)
     write(*,*) "ERROR in gmove_ge_s"
     call exit(1)
  end if
!$xmp end task

!$xmp end task

end subroutine gmove_ge_s
