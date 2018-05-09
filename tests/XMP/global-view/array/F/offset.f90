!$xmp nodes p(4)
!$xmp template t(20)
!$xmp distribute t(block) onto p

      integer a(14)
!$xmp align a(i) with t(i+2)

      integer b(20)
!$xmp align b(i) with t(i)

      integer a0(14), b0(20)

      integer me, xmp_node_num
      integer :: result = 0

      me = xmp_node_num()

! -------------------------------------------

      a0 = 1
      b0 = 0; b0(3:16) = 1
      
! -------------------------------------------

!$xmp array on t(3:16)
      a(1:14) = 0
!$xmp array on t
      b = 0

!$xmp loop on t(i+2)
      do i = 1, 14
         a(i) = 1
         b(i+2) = 1
      end do

!$xmp loop on t(i+2)
      do i = 1, 14
         if (a(i) /= a0(i)) result = -1
         if (b(i+2) /= b0(i+2)) result = -1
      end do

!$xmp reduction(+:result)

!$xmp task on p(1)
    if (result /= 0) then
       write(*,*) "ERROR"
       call exit(1)
       stop
    endif
!$xmp end task

! -------------------------------------------

!$xmp array on t(3:16)
      a(1:14) = 0
!$xmp array on t
      b = 0

!$xmp loop on t(i)
      do i = 1+2, 14+2
         a(i-2) = 1
         b(i) = 1
      end do

!$xmp loop on t(i+2)
      do i = 1, 14
         if (a(i) /= a0(i)) result = -1
         if (b(i+2) /= b0(i+2)) result = -1
      end do

!$xmp reduction(+:result)

!$xmp task on p(1)
    if (result /= 0) then
       write(*,*) "ERROR"
       call exit(1)
       stop
    endif
!$xmp end task

! -------------------------------------------

!$xmp array on t(3:16)
      a(1:14) = 0
!$xmp array on t
      b = 0

!$xmp array on t(3:16)
      a(1:14) = 1
!$xmp array on t(3:16)
      b(3:16) = 1

!$xmp loop on t(i+2)
      do i = 1, 14
         if (a(i) /= a0(i)) result = -1
         if (b(i+2) /= b0(i+2)) result = -1
      end do

!$xmp reduction(+:result)

!$xmp task on p(1)
    if (result /= 0) then
       write(*,*) "ERROR"
       call exit(1)
       stop
    endif
!$xmp end task

! -------------------------------------------

!$xmp task on p(1)
  write(*,*) "PASS"
!$xmp end task

    end program
