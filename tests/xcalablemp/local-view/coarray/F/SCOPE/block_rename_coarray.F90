#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
module m_block_rename_coarray_p
  integer a0_4[*]
  integer a0_14[*]
  integer a0_24[*]
  integer a0_1
  integer a0_7
  integer a0_23
end module
module m_block_rename_coarray
  use m_block_rename_coarray_p
  integer a0_2[*]
  integer a0_12[*]
  integer a0_22[*]
  integer a0_5
  integer a0_11
  integer a0_25
contains
  subroutine sub1
    integer, save :: a0_10[*]
    integer, save :: a0_20[*]
    integer a0_13
    integer a0_19
    a0_14 = 14
    a0_24 = 24
    a0_7  =  7
    a0_25 = 25
    a0_12 = 12
    a0_22 = 22
    a0_13 = 13
    a0_23 = 23
    a0_10 = 10
    a0_20 = 20
    a0_11 = 11
    a0_19 = 19
    call sub2
!   print *, 'a0_14 = ', a0_14
!   print *, 'a0_24 = ', a0_24
!   print *, 'a0_7  = ', a0_7
!   print *, 'a0_25 = ', a0_25
!   print *, 'a0_12 = ', a0_12
!   print *, 'a0_22 = ', a0_22
!   print *, 'a0_13 = ', a0_13
!   print *, 'a0_23 = ', a0_23
!   print *, 'a0_10 = ', a0_10
!   print *, 'a0_20 = ', a0_20
!   print *, 'a0_11 = ', a0_11
!   print *, 'a0_19 = ', a0_19
    if (a0_14.eq.14.and. &
        a0_24.eq.24.and. &
        a0_7 .eq. 7.and. &
        a0_25.eq.25.and. &
        a0_12.eq.12.and. &
        a0_22.eq.22.and. &
        a0_13.eq.13.and. &
        a0_23.eq.23.and. &
        a0_10.eq.10.and. &
        a0_20.eq.20.and. &
        a0_11.eq.11.and. &
        a0_19.eq.19     ) then
      print *, 'OK 3'
    else
      print *, 'NG 3'
      call exit(1)
    endif
  contains
    subroutine sub2
      integer, save :: a0_8[*]
      integer, save :: a0_18[*]
      integer a0_15
      integer a0_21
      a0_8 = 8
      a0_18 = 18
      a0_15 = 15
      a0_21 = 21
      block
        integer, save :: a0[*]
        integer a0_9
        integer a0_17
        a0 = 16
        a0_9 = 9
        a0_17 = 17
        block
          integer, save :: a0[*]
          a0 = 26
!         print *, 'a0_26 = ', a0
        end block
!       print *, 'a0_16 = ', a0
!       print *, 'a0_9  = ', a0_9
!       print *, 'a0_17 = ', a0_17
        if (a0   .eq.16.and. &
            a0_9 .eq. 9.and. &
            a0_17.eq.17     ) then
          print *, 'OK 1'
        else
          print *, 'NG 1'
          call exit(1)
        endif
      end block
!     print *, 'a0_8  = ', a0_8
!     print *, 'a0_18 = ', a0_18
!     print *, 'a0_15 = ', a0_15
!     print *, 'a0_21 = ', a0_21
      if (a0_8 .eq. 8.and. &
          a0_18.eq.18.and. &
          a0_15.eq.15.and. &
          a0_21.eq.21     ) then
        print *, 'OK 2'
      else
        print *, 'NG 2'
        call exit(1)
      endif
    end subroutine
  end subroutine
end module

  use m_block_rename_coarray

  integer a0_0[*]
  integer a0_3

  a0_0  = 0
  a0_1  = 1
  a0_2  = 2
  a0_3  = 3
  a0_4  = 4
  a0_5  = 5

  call sub1

  block
    integer, save :: a0[*]
    a0  = 6
!   print *, 'a0_6  = ', a0
  end block

! print *, 'a0_0  = ', a0_0
! print *, 'a0_1  = ', a0_1
! print *, 'a0_2  = ', a0_2
! print *, 'a0_3  = ', a0_3
! print *, 'a0_4  = ', a0_4
! print *, 'a0_5  = ', a0_5
  if (a0_0.eq.0.and. &
      a0_1.eq.1.and. &
      a0_2.eq.2.and. &
      a0_3.eq.3.and. &
      a0_4.eq.4.and. &
      a0_5.eq.5     ) then
    print *, 'OK 4'
  else
    print *, 'NG 2'
    call exit(1)
  endif

end
#else
print *, 'SKIPPED'
#endif

