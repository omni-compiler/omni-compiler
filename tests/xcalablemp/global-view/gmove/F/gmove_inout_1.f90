module mod0

  integer, parameter :: N = 8

!$xmp nodes p(2)

!$xmp template t1(N)
!$xmp distribute t1(block) onto p

  integer a(N)
!$xmp align a(i) with t1(i)
!$xmp shadow a(2:1)

!$xmp template t2(N)
!$xmp distribute t2(cyclic) onto p

  integer b(N)
!$xmp align b(i) with t2(i)

end module mod0


!--------------------------------------------------------
program test

  use mod0

#ifdef _MPI3
  call gmove_in
  call gmove_in_async
  call gmove_out
  call gmove_out_async

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task
#else
  write(*,*) "Skipped"
#endif

end program test


!--------------------------------------------------------
subroutine gmove_in

integer :: result = 0

  use mod0

!$xmp loop (i) on t1(i)
  do i = 1, N
     a(i) = 777
  end do

!$xmp loop (i) on t2(i)
  do i = 1, N
     b(i) = i
  end do

!$xmp barrier

#ifdef _MPI3
!$xmp gmove in
  a = b
#endif

!$xmp loop (i) on t1(i) reduction(+:result)
  do i = 1, N
     if (a(i) /= i) then
        result = 1
        !write(*,*) "(", xmp_node_num(), ")", i, a(i)
     end if
  end do

!$xmp task on p(1)
  if (result /= 0) then
     write(*,*) "ERROR in gmove_in"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_in


!--------------------------------------------------------
subroutine gmove_in_async

integer :: result = 0

  use mod0

!$xmp loop (i) on t1(i)
  do i = 1, N
     a(i) = 777
  end do

!$xmp loop (i) on t2(i)
  do i = 1, N
     b(i) = i
  end do

!$xmp barrier

#ifdef _MPI3
!$xmp gmove in async(10)
  a = b
#endif

!$xmp wait_async(10)

!$xmp loop (i) on t1(i) reduction(+:result)
  do i = 1, N
     if (a(i) /= i) then
        result = 1
        !write(*,*) "(", xmp_node_num(), ")", i, a(i)
     end if
  end do

!$xmp task on p(1)
  if (result /= 0) then
     write(*,*) "ERROR in gmove_in_async"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_in_async


!--------------------------------------------------------
subroutine gmove_out

integer :: result = 0

  use mod0

!$xmp loop (i) on t1(i)
  do i = 1, N
     a(i) = 777
  end do

!$xmp loop (i) on t2(i)
  do i = 1, N
     b(i) = i
  end do

!$xmp barrier

#ifdef _MPI3
!$xmp gmove out
  a = b
#endif

!$xmp loop (i) on t1(i) reduction(+:result)
  do i = 1, N
     if (a(i) /= i) then
        result = 1
        !write(*,*) "(", xmp_node_num(), ")", i, a(i)
     end if
  end do

!$xmp task on p(1)
  if (result /= 0) then
     write(*,*) "ERROR in gmove_out"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_out


!--------------------------------------------------------
subroutine gmove_out_async

integer :: result = 0

  use mod0

!$xmp loop (i) on t1(i)
  do i = 1, N
     a(i) = 777
  end do

!$xmp loop (i) on t2(i)
  do i = 1, N
     b(i) = i
  end do

!$xmp barrier

#ifdef _MPI3
!$xmp gmove out async(10)
  a = b
#endif

!$xmp wait_async(10)

!$xmp loop (i) on t1(i) reduction(+:result)
  do i = 1, N
     if (a(i) /= i) then
        result = 1
        !write(*,*) "(", xmp_node_num(), ")", i, a(i)
     end if
  end do

!$xmp task on p(1)
  if (result /= 0) then
     write(*,*) "ERROR in gmove_out_async"
     call exit(1)
  endif
!$xmp end task

end subroutine gmove_in_async
