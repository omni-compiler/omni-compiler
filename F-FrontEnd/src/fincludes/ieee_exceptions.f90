!    Implementation of the IEEE_EXCEPTIONS standard intrinsic module
!    Copyright (C) 2013-2017 Free Software Foundation, Inc.
!    Contributed by Francois-Xavier Coudert <fxcoudert@gcc.gnu.org>
!
! This file is part of the GNU Fortran runtime library (libgfortran).
!
! Libgfortran is free software; you can redistribute it and/or
! modify it under the terms of the GNU General Public
! License as published by the Free Software Foundation; either
! version 3 of the License, or (at your option) any later version.
!
! Libgfortran is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! Under Section 7 of GPL version 3, you are granted additional
! permissions described in the GCC Runtime Library Exception, version
! 3.1, as published by the Free Software Foundation.
!
! You should have received a copy of the GNU General Public License and
! a copy of the GCC Runtime Library Exception along with this program;
! see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
! <http://www.gnu.org/licenses/>.  */

!#include "config.h"
!#include "kinds.inc"
!#include "c99_protos.inc"
!#include "fpu-target.inc"

! Taken form libgfortran.h
#define GFC_FPE_INVALID      1
#define GFC_FPE_DENORMAL     2
#define GFC_FPE_ZERO         4
#define GFC_FPE_OVERFLOW     8
#define GFC_FPE_UNDERFLOW   16
#define GFC_FPE_INEXACT     32

module IEEE_EXCEPTIONS

  implicit none
  private

! Derived types and named constants

  type, public :: IEEE_FLAG_TYPE
    private
    integer :: hidden
  end type

  type(IEEE_FLAG_TYPE), parameter, public ::                 &
    IEEE_INVALID = IEEE_FLAG_TYPE(GFC_FPE_INVALID),          &
    IEEE_OVERFLOW       = IEEE_FLAG_TYPE(GFC_FPE_OVERFLOW),  &
    IEEE_DIVIDE_BY_ZERO = IEEE_FLAG_TYPE(GFC_FPE_ZERO),      &
    IEEE_UNDERFLOW      = IEEE_FLAG_TYPE(GFC_FPE_UNDERFLOW), &
    IEEE_INEXACT        = IEEE_FLAG_TYPE(GFC_FPE_INEXACT)

  type(IEEE_FLAG_TYPE), parameter, public :: &
     IEEE_USUAL(3) = [ IEEE_OVERFLOW, IEEE_DIVIDE_BY_ZERO, IEEE_INVALID ], &
     IEEE_ALL(5)   = [ IEEE_USUAL, IEEE_UNDERFLOW, IEEE_INEXACT ]

  type, public :: IEEE_STATUS_TYPE
    private
    character(len=*) :: hidden
  end type

  interface IEEE_SUPPORT_FLAG
    module procedure IEEE_SUPPORT_FLAG_4, &
                     IEEE_SUPPORT_FLAG_8, &
#ifdef HAVE_GFC_REAL_10
                     IEEE_SUPPORT_FLAG_10, &
#endif
#ifdef HAVE_GFC_REAL_16
                     IEEE_SUPPORT_FLAG_16, &
#endif
                     IEEE_SUPPORT_FLAG_NOARG
  end interface IEEE_SUPPORT_FLAG

  public :: IEEE_SUPPORT_FLAG, IEEE_SUPPORT_HALTING
  public :: IEEE_SET_HALTING_MODE, IEEE_GET_HALTING_MODE
  public :: IEEE_SET_FLAG, IEEE_GET_FLAG
  public :: IEEE_SET_STATUS, IEEE_GET_STATUS

contains

! Saving and restoring floating-point status

  subroutine IEEE_GET_STATUS (STATUS_VALUE)
    implicit none
    type(IEEE_STATUS_TYPE), intent(out) :: STATUS_VALUE
  end subroutine

  subroutine IEEE_SET_STATUS (STATUS_VALUE)
    implicit none
    type(IEEE_STATUS_TYPE), intent(in) :: STATUS_VALUE
  end subroutine

! Getting and setting flags

  elemental subroutine IEEE_GET_FLAG (FLAG, FLAG_VALUE)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    logical, intent(out) :: FLAG_VALUE
  end subroutine

  elemental subroutine IEEE_SET_FLAG (FLAG, FLAG_VALUE)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    logical, intent(in) :: FLAG_VALUE
  end subroutine

! Querying and changing the halting mode

  elemental subroutine IEEE_GET_HALTING_MODE (FLAG, HALTING)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    logical, intent(out) :: HALTING
  end subroutine

  elemental subroutine IEEE_SET_HALTING_MODE (FLAG, HALTING)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    logical, intent(in) :: HALTING
  end subroutine

! Querying support

  pure logical function IEEE_SUPPORT_HALTING (FLAG)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
  end function

  pure logical function IEEE_SUPPORT_FLAG_NOARG (FLAG)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
  end function

  pure logical function IEEE_SUPPORT_FLAG_4 (FLAG, X) result(res)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    real(kind=4), intent(in) :: X
    res = IEEE_SUPPORT_FLAG_NOARG(FLAG)
  end function

  pure logical function IEEE_SUPPORT_FLAG_8 (FLAG, X) result(res)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    real(kind=8), intent(in) :: X
    res = IEEE_SUPPORT_FLAG_NOARG(FLAG)
  end function

#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_FLAG_10 (FLAG, X) result(res)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    real(kind=10), intent(in) :: X
    res = IEEE_SUPPORT_FLAG_NOARG(FLAG)
  end function
#endif

#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_FLAG_16 (FLAG, X) result(res)
    implicit none
    type(IEEE_FLAG_TYPE), intent(in) :: FLAG
    real(kind=16), intent(in) :: X
    res = IEEE_SUPPORT_FLAG_NOARG(FLAG)
  end function
#endif

end module IEEE_EXCEPTIONS
