!    Implementation of the IEEE_ARITHMETIC standard intrinsic module
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

! Taken from libgfortran.h
! Defines for floating-point rounding modes.
#define GFC_FPE_DOWNWARD   1
#define GFC_FPE_TONEAREST  2
#define GFC_FPE_TOWARDZERO 3
#define GFC_FPE_UPWARD     4

module IEEE_ARITHMETIC

  use IEEE_EXCEPTIONS
  implicit none
  private

  ! Every public symbol from IEEE_EXCEPTIONS must be made public here
  public :: IEEE_FLAG_TYPE, IEEE_INVALID, IEEE_OVERFLOW, &
    IEEE_DIVIDE_BY_ZERO, IEEE_UNDERFLOW, IEEE_INEXACT, IEEE_USUAL, &
    IEEE_ALL, IEEE_STATUS_TYPE, IEEE_GET_FLAG, IEEE_GET_HALTING_MODE, &
    IEEE_GET_STATUS, IEEE_SET_FLAG, IEEE_SET_HALTING_MODE, &
    IEEE_SET_STATUS, IEEE_SUPPORT_FLAG, IEEE_SUPPORT_HALTING

  ! Derived types and named constants

  type, public :: IEEE_CLASS_TYPE
    private
    integer :: hidden
  end type

  type(IEEE_CLASS_TYPE), parameter, public :: &
    IEEE_OTHER_VALUE       = IEEE_CLASS_TYPE(0), &
    IEEE_SIGNALING_NAN     = IEEE_CLASS_TYPE(1), &
    IEEE_QUIET_NAN         = IEEE_CLASS_TYPE(2), &
    IEEE_NEGATIVE_INF      = IEEE_CLASS_TYPE(3), &
    IEEE_NEGATIVE_NORMAL   = IEEE_CLASS_TYPE(4), &
    IEEE_NEGATIVE_DENORMAL = IEEE_CLASS_TYPE(5), &
    IEEE_NEGATIVE_ZERO     = IEEE_CLASS_TYPE(6), &
    IEEE_POSITIVE_ZERO     = IEEE_CLASS_TYPE(7), &
    IEEE_POSITIVE_DENORMAL = IEEE_CLASS_TYPE(8), &
    IEEE_POSITIVE_NORMAL   = IEEE_CLASS_TYPE(9), &
    IEEE_POSITIVE_INF      = IEEE_CLASS_TYPE(10)

  type, public :: IEEE_ROUND_TYPE
    private
    integer :: hidden
  end type

  type(IEEE_ROUND_TYPE), parameter, public :: &
    IEEE_NEAREST           = IEEE_ROUND_TYPE(GFC_FPE_TONEAREST), &
    IEEE_TO_ZERO           = IEEE_ROUND_TYPE(GFC_FPE_TOWARDZERO), &
    IEEE_UP                = IEEE_ROUND_TYPE(GFC_FPE_UPWARD), &
    IEEE_DOWN              = IEEE_ROUND_TYPE(GFC_FPE_DOWNWARD), &
    IEEE_OTHER             = IEEE_ROUND_TYPE(0)


  ! Equality operators on the derived types
  interface operator (==)
    module procedure IEEE_CLASS_TYPE_EQ, IEEE_ROUND_TYPE_EQ
  end interface
  public :: operator(==)

  interface operator (/=)
    module procedure IEEE_CLASS_TYPE_NE, IEEE_ROUND_TYPE_NE
  end interface
  public :: operator (/=)


  ! IEEE_IS_FINITE

  interface
    elemental logical function gfortran_ieee_is_finite_4(X)
      real(kind=4), intent(in) :: X
    end function
    elemental logical function gfortran_ieee_is_finite_8(X)
      real(kind=8), intent(in) :: X
    end function
#ifdef HAVE_GFC_REAL_10
    elemental logical function gfortran_ieee_is_finite_10(X)
      real(kind=10), intent(in) :: X
    end function
#endif
#ifdef HAVE_GFC_REAL_16
    elemental logical function gfortran_ieee_is_finite_16(X)
      real(kind=16), intent(in) :: X
    end function
#endif
  end interface

  interface IEEE_IS_FINITE
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_is_finite_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_is_finite_10, &
#endif
      gfortran_ieee_is_finite_8, gfortran_ieee_is_finite_4
  end interface
  public :: IEEE_IS_FINITE

  ! IEEE_IS_NAN

  interface
    elemental logical function gfortran_ieee_is_nan_4(X)
      real(kind=4), intent(in) :: X
    end function
    elemental logical function gfortran_ieee_is_nan_8(X)
      real(kind=8), intent(in) :: X
    end function
#ifdef HAVE_GFC_REAL_10
    elemental logical function gfortran_ieee_is_nan_10(X)
      real(kind=10), intent(in) :: X
    end function
#endif
#ifdef HAVE_GFC_REAL_16
    elemental logical function gfortran_ieee_is_nan_16(X)
      real(kind=16), intent(in) :: X
    end function
#endif
  end interface

  interface IEEE_IS_NAN
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_is_nan_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_is_nan_10, &
#endif
      gfortran_ieee_is_nan_8, gfortran_ieee_is_nan_4
  end interface
  public :: IEEE_IS_NAN

  ! IEEE_IS_NEGATIVE

  interface
    elemental logical function gfortran_ieee_is_negative_4(X)
      real(kind=4), intent(in) :: X
    end function
    elemental logical function gfortran_ieee_is_negative_8(X)
      real(kind=8), intent(in) :: X
    end function
#ifdef HAVE_GFC_REAL_10
    elemental logical function gfortran_ieee_is_negative_10(X)
      real(kind=10), intent(in) :: X
    end function
#endif
#ifdef HAVE_GFC_REAL_16
    elemental logical function gfortran_ieee_is_negative_16(X)
      real(kind=16), intent(in) :: X
    end function
#endif
  end interface

  interface IEEE_IS_NEGATIVE
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_is_negative_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_is_negative_10, &
#endif
      gfortran_ieee_is_negative_8, gfortran_ieee_is_negative_4
  end interface
  public :: IEEE_IS_NEGATIVE

  ! IEEE_IS_NORMAL

  interface
    elemental logical function gfortran_ieee_is_normal_4(X)
      real(kind=4), intent(in) :: X
    end function
    elemental logical function gfortran_ieee_is_normal_8(X)
      real(kind=8), intent(in) :: X
    end function
#ifdef HAVE_GFC_REAL_10
    elemental logical function gfortran_ieee_is_normal_10(X)
      real(kind=10), intent(in) :: X
    end function
#endif
#ifdef HAVE_GFC_REAL_16
    elemental logical function gfortran_ieee_is_normal_16(X)
      real(kind=16), intent(in) :: X
    end function
#endif
  end interface

  interface IEEE_IS_NORMAL
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_is_normal_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_is_normal_10, &
#endif
      gfortran_ieee_is_normal_8, gfortran_ieee_is_normal_4
  end interface
  public :: IEEE_IS_NORMAL

  ! IEEE_COPY_SIGN
  interface
    elemental real(kind = 4) function gfortran_ieee_copy_sign_4_4 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 4), intent(in) :: Y
    end function
    elemental real(kind = 4) function gfortran_ieee_copy_sign_4_8 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 8), intent(in) :: Y
    end function

#ifdef HAVE_GFC_REAL_10
    elemental real(kind = 4) function gfortran_ieee_copy_sign_4_10 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 10), intent(in) :: Y
    end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 4) function gfortran_ieee_copy_sign_4_16 (X,Y)
    real(kind = 4), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  elemental real(kind = 8) function gfortran_ieee_copy_sign_8_4 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 8) function gfortran_ieee_copy_sign_8_8 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 8) function gfortran_ieee_copy_sign_8_10 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 8) function gfortran_ieee_copy_sign_8_16 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 10) function gfortran_ieee_copy_sign_10_4 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 10) function gfortran_ieee_copy_sign_10_8 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
  elemental real(kind = 10) function gfortran_ieee_copy_sign_10_10 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 10) function gfortran_ieee_copy_sign_10_16 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 16) function gfortran_ieee_copy_sign_16_4 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 16) function gfortran_ieee_copy_sign_16_8 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 16) function gfortran_ieee_copy_sign_16_10 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
  elemental real(kind = 16) function gfortran_ieee_copy_sign_16_16 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  end interface

  interface IEEE_COPY_SIGN
    procedure &
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_copy_sign_16_16, &
#ifdef HAVE_GFC_REAL_10
              gfortran_ieee_copy_sign_16_10, &
#endif
              gfortran_ieee_copy_sign_16_8, &
              gfortran_ieee_copy_sign_16_4, &
#endif
#ifdef HAVE_GFC_REAL_10
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_copy_sign_10_16, &
#endif
              gfortran_ieee_copy_sign_10_10, &
              gfortran_ieee_copy_sign_10_8, &
              gfortran_ieee_copy_sign_10_4, &
#endif
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_copy_sign_8_16, &
#endif
#ifdef HAVE_GFC_REAL_10
              gfortran_ieee_copy_sign_8_10, &
#endif
              gfortran_ieee_copy_sign_8_8, &
              gfortran_ieee_copy_sign_8_4, &
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_copy_sign_4_16, &
#endif
#ifdef HAVE_GFC_REAL_10
              gfortran_ieee_copy_sign_4_10, &
#endif
              gfortran_ieee_copy_sign_4_8, &
              gfortran_ieee_copy_sign_4_4
  end interface
  public :: IEEE_COPY_SIGN

  ! IEEE_UNORDERED
  interface
    elemental logical function gfortran_ieee_unordered_4_4 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 4), intent(in) :: Y
    end function
    elemental logical function gfortran_ieee_unordered_4_8 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 8), intent(in) :: Y
    end function
#ifdef HAVE_GFC_REAL_10
  elemental logical function gfortran_ieee_unordered_4_10 (X,Y)
    real(kind = 4), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental logical function gfortran_ieee_unordered_4_16 (X,Y)
    real(kind = 4), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  elemental logical function gfortran_ieee_unordered_8_4 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental logical function gfortran_ieee_unordered_8_8 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
  elemental logical function gfortran_ieee_unordered_8_10 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental logical function gfortran_ieee_unordered_8_16 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_10
  elemental logical function gfortran_ieee_unordered_10_4 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental logical function gfortran_ieee_unordered_10_8 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
  elemental logical function gfortran_ieee_unordered_10_10 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_16
  elemental logical function gfortran_ieee_unordered_10_16 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#endif
#ifdef HAVE_GFC_REAL_16
  elemental logical function gfortran_ieee_unordered_16_4 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental logical function gfortran_ieee_unordered_16_8 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
  elemental logical function gfortran_ieee_unordered_16_10 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
  elemental logical function gfortran_ieee_unordered_16_16 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  end interface

  interface IEEE_UNORDERED
    procedure &
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_unordered_16_16, &
#ifdef HAVE_GFC_REAL_10
              gfortran_ieee_unordered_16_10, &
#endif
              gfortran_ieee_unordered_16_8, &
              gfortran_ieee_unordered_16_4, &
#endif
#ifdef HAVE_GFC_REAL_10
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_unordered_10_16, &
#endif
              gfortran_ieee_unordered_10_10, &
              gfortran_ieee_unordered_10_8, &
              gfortran_ieee_unordered_10_4, &
#endif
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_unordered_8_16, &
#endif
#ifdef HAVE_GFC_REAL_10
              gfortran_ieee_unordered_8_10, &
#endif
              gfortran_ieee_unordered_8_8, &
              gfortran_ieee_unordered_8_4, &
#ifdef HAVE_GFC_REAL_16
              gfortran_ieee_unordered_4_16, &
#endif
#ifdef HAVE_GFC_REAL_10
              gfortran_ieee_unordered_4_10, &
#endif
              gfortran_ieee_unordered_4_8, &
              gfortran_ieee_unordered_4_4
  end interface
  public :: IEEE_UNORDERED

  ! IEEE_LOGB

  interface
    elemental real(kind=4) function gfortran_ieee_logb_4 (X)
      real(kind=4), intent(in) :: X
    end function
    elemental real(kind=8) function gfortran_ieee_logb_8 (X)
      real(kind=8), intent(in) :: X
    end function
#ifdef HAVE_GFC_REAL_10
    elemental real(kind=10) function gfortran_ieee_logb_10 (X)
      real(kind=10), intent(in) :: X
    end function
#endif
#ifdef HAVE_GFC_REAL_16
    elemental real(kind=16) function gfortran_ieee_logb_16 (X)
      real(kind=16), intent(in) :: X
    end function
#endif
  end interface

  interface IEEE_LOGB
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_logb_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_logb_10, &
#endif
      gfortran_ieee_logb_8, &
      gfortran_ieee_logb_4
  end interface
  public :: IEEE_LOGB

  ! IEEE_NEXT_AFTER
  interface
    elemental real(kind = 4) function gfortran_ieee_next_after_4_4 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 4), intent(in) :: Y
    end function
    elemental real(kind = 4) function gfortran_ieee_next_after_4_8 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 8), intent(in) :: Y
    end function
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 4) function gfortran_ieee_next_after_4_10 (X,Y)
    real(kind = 4), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 4) function gfortran_ieee_next_after_4_16 (X,Y)
    real(kind = 4), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  elemental real(kind = 8) function gfortran_ieee_next_after_8_4 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 8) function gfortran_ieee_next_after_8_8 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 8) function gfortran_ieee_next_after_8_10 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 8) function gfortran_ieee_next_after_8_16 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 10) function gfortran_ieee_next_after_10_4 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 10) function gfortran_ieee_next_after_10_8 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
  elemental real(kind = 10) function gfortran_ieee_next_after_10_10 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 10) function gfortran_ieee_next_after_10_16 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 16) function gfortran_ieee_next_after_16_4 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 16) function gfortran_ieee_next_after_16_8 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 16) function gfortran_ieee_next_after_16_10 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
  elemental real(kind = 16) function gfortran_ieee_next_after_16_16 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  end interface

  interface IEEE_NEXT_AFTER
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_next_after_16_16, &
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_next_after_16_10, &
#endif
      gfortran_ieee_next_after_16_8, &
      gfortran_ieee_next_after_16_4, &
#endif
#ifdef HAVE_GFC_REAL_10
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_next_after_10_16, &
#endif
      gfortran_ieee_next_after_10_10, &
      gfortran_ieee_next_after_10_8, &
      gfortran_ieee_next_after_10_4, &
#endif
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_next_after_8_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_next_after_8_10, &
#endif
      gfortran_ieee_next_after_8_8, &
      gfortran_ieee_next_after_8_4, &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_next_after_4_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_next_after_4_10, &
#endif
      gfortran_ieee_next_after_4_8, &
      gfortran_ieee_next_after_4_4
  end interface
  public :: IEEE_NEXT_AFTER

  ! IEEE_REM
  interface
    elemental real(kind = 4) function gfortran_ieee_rem_4_4 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 4), intent(in) :: Y
    end function
    elemental real(kind = 8) function gfortran_ieee_rem_4_8 (X,Y)
      real(kind = 4), intent(in) :: X
      real(kind = 8), intent(in) :: Y
    end function
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 10) function gfortran_ieee_rem_4_10 (X,Y)
    real(kind = 4), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 16) function gfortran_ieee_rem_4_16 (X,Y)
    real(kind = 4), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  elemental real(kind = 8) function gfortran_ieee_rem_8_4 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 8) function gfortran_ieee_rem_8_8 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
elemental real(kind = 10) function gfortran_ieee_rem_8_10 (X,Y)
  real(kind = 8), intent(in) :: X
  real(kind = 10), intent(in) :: Y
end function
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 16) function gfortran_ieee_rem_8_16 (X,Y)
    real(kind = 8), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 10) function gfortran_ieee_rem_10_4 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 10) function gfortran_ieee_rem_10_8 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
  elemental real(kind = 10) function gfortran_ieee_rem_10_10 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 10) function gfortran_ieee_rem_10_16 (X,Y)
    real(kind = 10), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
#endif
#ifdef HAVE_GFC_REAL_16
  elemental real(kind = 16) function gfortran_ieee_rem_16_4 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 4), intent(in) :: Y
  end function
  elemental real(kind = 16) function gfortran_ieee_rem_16_8 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 8), intent(in) :: Y
  end function
#ifdef HAVE_GFC_REAL_10
  elemental real(kind = 16) function gfortran_ieee_rem_16_10 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 10), intent(in) :: Y
  end function
#endif
  elemental real(kind = 16) function gfortran_ieee_rem_16_16 (X,Y)
    real(kind = 16), intent(in) :: X
    real(kind = 16), intent(in) :: Y
  end function
#endif
  end interface

  interface IEEE_REM
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_rem_16_16, &
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_rem_16_10, &
#endif
      gfortran_ieee_rem_16_8, &
      gfortran_ieee_rem_16_4, &
#endif
#ifdef HAVE_GFC_REAL_10
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_rem_10_16, &
#endif
      gfortran_ieee_rem_10_10, &
      gfortran_ieee_rem_10_8, &
      gfortran_ieee_rem_10_4, &
#endif
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_rem_8_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_rem_8_10, &
#endif
      gfortran_ieee_rem_8_8, &
      gfortran_ieee_rem_8_4, &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_rem_4_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_rem_4_10, &
#endif
      gfortran_ieee_rem_4_8, &
      gfortran_ieee_rem_4_4
  end interface
  public :: IEEE_REM

  ! IEEE_RINT

  interface
    elemental real(kind=4) function gfortran_ieee_rint_4 (X)
      real(kind=4), intent(in) :: X
    end function
    elemental real(kind=8) function gfortran_ieee_rint_8 (X)
      real(kind=8), intent(in) :: X
    end function
#ifdef HAVE_GFC_REAL_10
    elemental real(kind=10) function gfortran_ieee_rint_10 (X)
      real(kind=10), intent(in) :: X
    end function
#endif
#ifdef HAVE_GFC_REAL_16
    elemental real(kind=16) function gfortran_ieee_rint_16 (X)
      real(kind=16), intent(in) :: X
    end function
#endif
  end interface

  interface IEEE_RINT
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_rint_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_rint_10, &
#endif
      gfortran_ieee_rint_8, gfortran_ieee_rint_4
  end interface
  public :: IEEE_RINT

  ! IEEE_SCALB

  interface
    elemental real(kind=4) function gfortran_ieee_scalb_4 (X, I)
      real(kind=4), intent(in) :: X
      integer, intent(in) :: I
    end function
    elemental real(kind=8) function gfortran_ieee_scalb_8 (X, I)
      real(kind=8), intent(in) :: X
      integer, intent(in) :: I
    end function
#ifdef HAVE_GFC_REAL_10
    elemental real(kind=10) function gfortran_ieee_scalb_10 (X, I)
      real(kind=10), intent(in) :: X
      integer, intent(in) :: I
    end function
#endif
#ifdef HAVE_GFC_REAL_16
    elemental real(kind=16) function gfortran_ieee_scalb_16 (X, I)
      real(kind=16), intent(in) :: X
      integer, intent(in) :: I
    end function
#endif
  end interface

  interface IEEE_SCALB
    procedure &
#ifdef HAVE_GFC_REAL_16
      gfortran_ieee_scalb_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      gfortran_ieee_scalb_10, &
#endif
      gfortran_ieee_scalb_8, gfortran_ieee_scalb_4
  end interface
  public :: IEEE_SCALB

  ! IEEE_VALUE

  interface IEEE_VALUE
    module procedure &
#ifdef HAVE_GFC_REAL_16
      IEEE_VALUE_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      IEEE_VALUE_10, &
#endif
      IEEE_VALUE_8, IEEE_VALUE_4
  end interface
  public :: IEEE_VALUE

  ! IEEE_CLASS

  interface IEEE_CLASS
    module procedure &
#ifdef HAVE_GFC_REAL_16
      IEEE_CLASS_16, &
#endif
#ifdef HAVE_GFC_REAL_10
      IEEE_CLASS_10, &
#endif
      IEEE_CLASS_8, IEEE_CLASS_4
  end interface
  public :: IEEE_CLASS

  ! Public declarations for contained procedures
  public :: IEEE_GET_ROUNDING_MODE, IEEE_SET_ROUNDING_MODE
  public :: IEEE_GET_UNDERFLOW_MODE, IEEE_SET_UNDERFLOW_MODE
  public :: IEEE_SELECTED_REAL_KIND

  ! IEEE_SUPPORT_ROUNDING

  interface IEEE_SUPPORT_ROUNDING
    module procedure IEEE_SUPPORT_ROUNDING_4, IEEE_SUPPORT_ROUNDING_8, &
#ifdef HAVE_GFC_REAL_10
                     IEEE_SUPPORT_ROUNDING_10, &
#endif
#ifdef HAVE_GFC_REAL_16
                     IEEE_SUPPORT_ROUNDING_16, &
#endif
                     IEEE_SUPPORT_ROUNDING_NOARG
  end interface
  public :: IEEE_SUPPORT_ROUNDING

  ! Interface to the FPU-specific function
  interface
    pure integer function support_rounding_helper(flag) &
        bind(c, name="_gfortrani_support_fpu_rounding_mode")
      integer, intent(in), value :: flag
    end function
  end interface

  ! IEEE_SUPPORT_UNDERFLOW_CONTROL

  interface IEEE_SUPPORT_UNDERFLOW_CONTROL
    module procedure IEEE_SUPPORT_UNDERFLOW_CONTROL_4, &
                     IEEE_SUPPORT_UNDERFLOW_CONTROL_8, &
#ifdef HAVE_GFC_REAL_10
                     IEEE_SUPPORT_UNDERFLOW_CONTROL_10, &
#endif
#ifdef HAVE_GFC_REAL_16
                     IEEE_SUPPORT_UNDERFLOW_CONTROL_16, &
#endif
                     IEEE_SUPPORT_UNDERFLOW_CONTROL_NOARG
  end interface
  public :: IEEE_SUPPORT_UNDERFLOW_CONTROL

  ! Interface to the FPU-specific function
  interface
    pure integer function support_underflow_control_helper(kind) &
        bind(c, name="_gfortrani_support_fpu_underflow_control")
      integer, intent(in), value :: kind
    end function
  end interface

! IEEE_SUPPORT_* generic functions
#if defined(HAVE_GFC_REAL_10) && defined(HAVE_GFC_REAL_16)
  interface IEEE_SUPPORT_DATATYPE ; module procedure IEEE_SUPPORT_DATATYPE_4, IEEE_SUPPORT_DATATYPE_8, IEEE_SUPPORT_DATATYPE_10, IEEE_SUPPORT_DATATYPE_16, IEEE_SUPPORT_DATATYPE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DATATYPE
  interface IEEE_SUPPORT_DENORMAL ; module procedure IEEE_SUPPORT_DENORMAL_4, IEEE_SUPPORT_DENORMAL_8, IEEE_SUPPORT_DENORMAL_10, IEEE_SUPPORT_DENORMAL_16, IEEE_SUPPORT_DENORMAL_NOARG ; end interface ;   public :: IEEE_SUPPORT_DENORMAL
  interface IEEE_SUPPORT_DIVIDE ; module procedure IEEE_SUPPORT_DIVIDE_4, IEEE_SUPPORT_DIVIDE_8, IEEE_SUPPORT_DIVIDE_10, IEEE_SUPPORT_DIVIDE_16, IEEE_SUPPORT_DIVIDE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DIVIDE
  interface IEEE_SUPPORT_INF ; module procedure IEEE_SUPPORT_INF_4, IEEE_SUPPORT_INF_8, IEEE_SUPPORT_INF_10, IEEE_SUPPORT_INF_16, IEEE_SUPPORT_INF_NOARG ; end interface ;   public :: IEEE_SUPPORT_INF
  interface IEEE_SUPPORT_IO ; module procedure IEEE_SUPPORT_IO_4, IEEE_SUPPORT_IO_8, IEEE_SUPPORT_IO_10, IEEE_SUPPORT_IO_16, IEEE_SUPPORT_IO_NOARG ; end interface ;   public :: IEEE_SUPPORT_IO
  interface IEEE_SUPPORT_NAN ; module procedure IEEE_SUPPORT_NAN_4, IEEE_SUPPORT_NAN_8, IEEE_SUPPORT_NAN_10, IEEE_SUPPORT_NAN_16, IEEE_SUPPORT_NAN_NOARG ; end interface ;   public :: IEEE_SUPPORT_NAN
  interface IEEE_SUPPORT_SQRT ; module procedure IEEE_SUPPORT_SQRT_4, IEEE_SUPPORT_SQRT_8, IEEE_SUPPORT_SQRT_10, IEEE_SUPPORT_SQRT_16, IEEE_SUPPORT_SQRT_NOARG ; end interface ;   public :: IEEE_SUPPORT_SQRT
  interface IEEE_SUPPORT_STANDARD ; module procedure IEEE_SUPPORT_STANDARD_4, IEEE_SUPPORT_STANDARD_8, IEEE_SUPPORT_STANDARD_10, IEEE_SUPPORT_STANDARD_16, IEEE_SUPPORT_STANDARD_NOARG ; end interface ;   public :: IEEE_SUPPORT_STANDARD
#elif defined(HAVE_GFC_REAL_10)
  interface IEEE_SUPPORT_DATATYPE ; module procedure IEEE_SUPPORT_DATATYPE_4, IEEE_SUPPORT_DATATYPE_8, IEEE_SUPPORT_DATATYPE_10, IEEE_SUPPORT_DATATYPE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DATATYPE
  interface IEEE_SUPPORT_DENORMAL ; module procedure IEEE_SUPPORT_DENORMAL_4, IEEE_SUPPORT_DENORMAL_8, IEEE_SUPPORT_DENORMAL_10, IEEE_SUPPORT_DENORMAL_NOARG ; end interface ;   public :: IEEE_SUPPORT_DENORMAL
  interface IEEE_SUPPORT_DIVIDE ; module procedure IEEE_SUPPORT_DIVIDE_4, IEEE_SUPPORT_DIVIDE_8, IEEE_SUPPORT_DIVIDE_10, IEEE_SUPPORT_DIVIDE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DIVIDE
  interface IEEE_SUPPORT_INF ; module procedure IEEE_SUPPORT_INF_4, IEEE_SUPPORT_INF_8, IEEE_SUPPORT_INF_10, IEEE_SUPPORT_INF_NOARG ; end interface ;   public :: IEEE_SUPPORT_INF
  interface IEEE_SUPPORT_IO ; module procedure IEEE_SUPPORT_IO_4, IEEE_SUPPORT_IO_8, IEEE_SUPPORT_IO_10, IEEE_SUPPORT_IO_NOARG ; end interface ;   public :: IEEE_SUPPORT_IO
  interface IEEE_SUPPORT_NAN ; module procedure IEEE_SUPPORT_NAN_4, IEEE_SUPPORT_NAN_8, IEEE_SUPPORT_NAN_10, IEEE_SUPPORT_NAN_NOARG ; end interface ;   public :: IEEE_SUPPORT_NAN
  interface IEEE_SUPPORT_SQRT ; module procedure IEEE_SUPPORT_SQRT_4, IEEE_SUPPORT_SQRT_8, IEEE_SUPPORT_SQRT_10, IEEE_SUPPORT_SQRT_NOARG ; end interface ;   public :: IEEE_SUPPORT_SQRT
  interface IEEE_SUPPORT_STANDARD ; module procedure IEEE_SUPPORT_STANDARD_4, IEEE_SUPPORT_STANDARD_8, IEEE_SUPPORT_STANDARD_10, IEEE_SUPPORT_STANDARD_NOARG ; end interface ;   public :: IEEE_SUPPORT_STANDARD
#elif defined(HAVE_GFC_REAL_16)
  interface IEEE_SUPPORT_DATATYPE ; module procedure IEEE_SUPPORT_DATATYPE_4, IEEE_SUPPORT_DATATYPE_8, IEEE_SUPPORT_DATATYPE_16, IEEE_SUPPORT_DATATYPE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DATATYPE
  interface IEEE_SUPPORT_DENORMAL ; module procedure IEEE_SUPPORT_DENORMAL_4, IEEE_SUPPORT_DENORMAL_8, IEEE_SUPPORT_DENORMAL_16, IEEE_SUPPORT_DENORMAL_NOARG ; end interface ;   public :: IEEE_SUPPORT_DENORMAL
  interface IEEE_SUPPORT_DIVIDE ; module procedure IEEE_SUPPORT_DIVIDE_4, IEEE_SUPPORT_DIVIDE_8, IEEE_SUPPORT_DIVIDE_16, IEEE_SUPPORT_DIVIDE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DIVIDE
  interface IEEE_SUPPORT_INF ; module procedure IEEE_SUPPORT_INF_4, IEEE_SUPPORT_INF_8, IEEE_SUPPORT_INF_16, IEEE_SUPPORT_INF_NOARG ; end interface ;   public :: IEEE_SUPPORT_INF
  interface IEEE_SUPPORT_IO ; module procedure IEEE_SUPPORT_IO_4, IEEE_SUPPORT_IO_8, IEEE_SUPPORT_IO_16, IEEE_SUPPORT_IO_NOARG ; end interface ;   public :: IEEE_SUPPORT_IO
  interface IEEE_SUPPORT_NAN ; module procedure IEEE_SUPPORT_NAN_4, IEEE_SUPPORT_NAN_8, IEEE_SUPPORT_NAN_16, IEEE_SUPPORT_NAN_NOARG ; end interface ;   public :: IEEE_SUPPORT_NAN
  interface IEEE_SUPPORT_SQRT ; module procedure IEEE_SUPPORT_SQRT_4, IEEE_SUPPORT_SQRT_8, IEEE_SUPPORT_SQRT_16, IEEE_SUPPORT_SQRT_NOARG ; end interface ;   public :: IEEE_SUPPORT_SQRT
  interface IEEE_SUPPORT_STANDARD ; module procedure IEEE_SUPPORT_STANDARD_4, IEEE_SUPPORT_STANDARD_8, IEEE_SUPPORT_STANDARD_16, IEEE_SUPPORT_STANDARD_NOARG ; end interface ;   public :: IEEE_SUPPORT_STANDARD
#else
  interface IEEE_SUPPORT_DATATYPE ; module procedure IEEE_SUPPORT_DATATYPE_4, IEEE_SUPPORT_DATATYPE_8, IEEE_SUPPORT_DATATYPE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DATATYPE
  interface IEEE_SUPPORT_DENORMAL ; module procedure IEEE_SUPPORT_DENORMAL_4, IEEE_SUPPORT_DENORMAL_8, IEEE_SUPPORT_DENORMAL_NOARG ; end interface ;   public :: IEEE_SUPPORT_DENORMAL
  interface IEEE_SUPPORT_DIVIDE ; module procedure IEEE_SUPPORT_DIVIDE_4, IEEE_SUPPORT_DIVIDE_8, IEEE_SUPPORT_DIVIDE_NOARG ; end interface ;   public :: IEEE_SUPPORT_DIVIDE
  interface IEEE_SUPPORT_INF ; module procedure IEEE_SUPPORT_INF_4, IEEE_SUPPORT_INF_8, IEEE_SUPPORT_INF_NOARG ; end interface ;   public :: IEEE_SUPPORT_INF
  interface IEEE_SUPPORT_IO ; module procedure IEEE_SUPPORT_IO_4, IEEE_SUPPORT_IO_8, IEEE_SUPPORT_IO_NOARG ; end interface ;   public :: IEEE_SUPPORT_IO
  interface IEEE_SUPPORT_NAN ; module procedure IEEE_SUPPORT_NAN_4, IEEE_SUPPORT_NAN_8, IEEE_SUPPORT_NAN_NOARG ; end interface ;   public :: IEEE_SUPPORT_NAN
  interface IEEE_SUPPORT_SQRT ; module procedure IEEE_SUPPORT_SQRT_4, IEEE_SUPPORT_SQRT_8, IEEE_SUPPORT_SQRT_NOARG ; end interface ;   public :: IEEE_SUPPORT_SQRT
  interface IEEE_SUPPORT_STANDARD ; module procedure IEEE_SUPPORT_STANDARD_4, IEEE_SUPPORT_STANDARD_8, IEEE_SUPPORT_STANDARD_NOARG ; end interface ;   public :: IEEE_SUPPORT_STANDARD
#endif

contains

  ! Equality operators for IEEE_CLASS_TYPE and IEEE_ROUNDING_MODE
  elemental logical function IEEE_CLASS_TYPE_EQ (X, Y) result(res)
    implicit none
    type(IEEE_CLASS_TYPE), intent(in) :: X, Y
    res = (X%hidden == Y%hidden)
  end function

  elemental logical function IEEE_CLASS_TYPE_NE (X, Y) result(res)
    implicit none
    type(IEEE_CLASS_TYPE), intent(in) :: X, Y
    res = (X%hidden /= Y%hidden)
  end function

  elemental logical function IEEE_ROUND_TYPE_EQ (X, Y) result(res)
    implicit none
    type(IEEE_ROUND_TYPE), intent(in) :: X, Y
    res = (X%hidden == Y%hidden)
  end function

  elemental logical function IEEE_ROUND_TYPE_NE (X, Y) result(res)
    implicit none
    type(IEEE_ROUND_TYPE), intent(in) :: X, Y
    res = (X%hidden /= Y%hidden)
  end function


  ! IEEE_SELECTED_REAL_KIND

  integer function IEEE_SELECTED_REAL_KIND (P, R, RADIX) result(res)
    implicit none
    integer, intent(in), optional :: P, R, RADIX
    ! Only signature needed for OMNI
  end function


  ! IEEE_CLASS

  elemental function IEEE_CLASS_4 (X) result(res)
    implicit none
    real(kind=4), intent(in) :: X
    type(IEEE_CLASS_TYPE) :: res
    ! Only signature needed for OMNI
  end function

  elemental function IEEE_CLASS_8 (X) result(res)
    implicit none
    real(kind=8), intent(in) :: X
    type(IEEE_CLASS_TYPE) :: res
    ! Only signature needed for OMNI
  end function

#ifdef HAVE_GFC_REAL_10
  elemental function IEEE_CLASS_10 (X) result(res)
    implicit none
    real(kind=10), intent(in) :: X
    type(IEEE_CLASS_TYPE) :: res
    ! Only signature needed for OMNI
  end function
#endif

#ifdef HAVE_GFC_REAL_16
  elemental function IEEE_CLASS_16 (X) result(res)
    implicit none
    real(kind=16), intent(in) :: X
    type(IEEE_CLASS_TYPE) :: res
    ! Only signature needed for OMNI
  end function
#endif


  ! IEEE_VALUE

  elemental real(kind=4) function IEEE_VALUE_4(X, CLASS) result(res)

    real(kind=4), intent(in) :: X
    type(IEEE_CLASS_TYPE), intent(in) :: CLASS

    select case (CLASS%hidden)
      case (1)     ! IEEE_SIGNALING_NAN
        res = -1
        res = sqrt(res)
      case (2)     ! IEEE_QUIET_NAN
        res = -1
        res = sqrt(res)
      case (3)     ! IEEE_NEGATIVE_INF
        res = huge(res)
        res = (-res) * res
      case (4)     ! IEEE_NEGATIVE_NORMAL
        res = -42
      case (5)     ! IEEE_NEGATIVE_DENORMAL
        res = -tiny(res)
        res = res / 2
      case (6)     ! IEEE_NEGATIVE_ZERO
        res = 0
        res = -res
      case (7)     ! IEEE_POSITIVE_ZERO
        res = 0
      case (8)     ! IEEE_POSITIVE_DENORMAL
        res = tiny(res)
        res = res / 2
      case (9)     ! IEEE_POSITIVE_NORMAL
        res = 42
      case (10)    ! IEEE_POSITIVE_INF
        res = huge(res)
        res = res * res
      case default ! IEEE_OTHER_VALUE, should not happen
        res = 0
     end select
  end function

  elemental real(kind=8) function IEEE_VALUE_8(X, CLASS) result(res)

    real(kind=8), intent(in) :: X
    type(IEEE_CLASS_TYPE), intent(in) :: CLASS

    select case (CLASS%hidden)
      case (1)     ! IEEE_SIGNALING_NAN
        res = -1
        res = sqrt(res)
      case (2)     ! IEEE_QUIET_NAN
        res = -1
        res = sqrt(res)
      case (3)     ! IEEE_NEGATIVE_INF
        res = huge(res)
        res = (-res) * res
      case (4)     ! IEEE_NEGATIVE_NORMAL
        res = -42
      case (5)     ! IEEE_NEGATIVE_DENORMAL
        res = -tiny(res)
        res = res / 2
      case (6)     ! IEEE_NEGATIVE_ZERO
        res = 0
        res = -res
      case (7)     ! IEEE_POSITIVE_ZERO
        res = 0
      case (8)     ! IEEE_POSITIVE_DENORMAL
        res = tiny(res)
        res = res / 2
      case (9)     ! IEEE_POSITIVE_NORMAL
        res = 42
      case (10)    ! IEEE_POSITIVE_INF
        res = huge(res)
        res = res * res
      case default ! IEEE_OTHER_VALUE, should not happen
        res = 0
     end select
  end function

#ifdef HAVE_GFC_REAL_10
  elemental real(kind=10) function IEEE_VALUE_10(X, CLASS) result(res)

    real(kind=10), intent(in) :: X
    type(IEEE_CLASS_TYPE), intent(in) :: CLASS

    select case (CLASS%hidden)
      case (1)     ! IEEE_SIGNALING_NAN
        res = -1
        res = sqrt(res)
      case (2)     ! IEEE_QUIET_NAN
        res = -1
        res = sqrt(res)
      case (3)     ! IEEE_NEGATIVE_INF
        res = huge(res)
        res = (-res) * res
      case (4)     ! IEEE_NEGATIVE_NORMAL
        res = -42
      case (5)     ! IEEE_NEGATIVE_DENORMAL
        res = -tiny(res)
        res = res / 2
      case (6)     ! IEEE_NEGATIVE_ZERO
        res = 0
        res = -res
      case (7)     ! IEEE_POSITIVE_ZERO
        res = 0
      case (8)     ! IEEE_POSITIVE_DENORMAL
        res = tiny(res)
        res = res / 2
      case (9)     ! IEEE_POSITIVE_NORMAL
        res = 42
      case (10)    ! IEEE_POSITIVE_INF
        res = huge(res)
        res = res * res
      case default ! IEEE_OTHER_VALUE, should not happen
        res = 0
     end select
  end function

#endif

#ifdef HAVE_GFC_REAL_16
  elemental real(kind=16) function IEEE_VALUE_16(X, CLASS) result(res)

    real(kind=16), intent(in) :: X
    type(IEEE_CLASS_TYPE), intent(in) :: CLASS

    select case (CLASS%hidden)
      case (1)     ! IEEE_SIGNALING_NAN
        res = -1
        res = sqrt(res)
      case (2)     ! IEEE_QUIET_NAN
        res = -1
        res = sqrt(res)
      case (3)     ! IEEE_NEGATIVE_INF
        res = huge(res)
        res = (-res) * res
      case (4)     ! IEEE_NEGATIVE_NORMAL
        res = -42
      case (5)     ! IEEE_NEGATIVE_DENORMAL
        res = -tiny(res)
        res = res / 2
      case (6)     ! IEEE_NEGATIVE_ZERO
        res = 0
        res = -res
      case (7)     ! IEEE_POSITIVE_ZERO
        res = 0
      case (8)     ! IEEE_POSITIVE_DENORMAL
        res = tiny(res)
        res = res / 2
      case (9)     ! IEEE_POSITIVE_NORMAL
        res = 42
      case (10)    ! IEEE_POSITIVE_INF
        res = huge(res)
        res = res * res
      case default ! IEEE_OTHER_VALUE, should not happen
        res = 0
     end select
  end function
#endif


  ! IEEE_GET_ROUNDING_MODE

  subroutine IEEE_GET_ROUNDING_MODE (ROUND_VALUE)
    implicit none
    type(IEEE_ROUND_TYPE), intent(out) :: ROUND_VALUE
    ! Only signature needed for OMNI
  end subroutine


  ! IEEE_SET_ROUNDING_MODE

  subroutine IEEE_SET_ROUNDING_MODE (ROUND_VALUE)
    implicit none
    type(IEEE_ROUND_TYPE), intent(in) :: ROUND_VALUE
    ! Only signature needed for OMNI
  end subroutine


  ! IEEE_GET_UNDERFLOW_MODE

  subroutine IEEE_GET_UNDERFLOW_MODE (GRADUAL)
    implicit none
    logical, intent(out) :: GRADUAL
    ! Only signature needed for OMNI
  end subroutine


  ! IEEE_SET_UNDERFLOW_MODE

  subroutine IEEE_SET_UNDERFLOW_MODE (GRADUAL)
    implicit none
    logical, intent(in) :: GRADUAL
    ! Only signature needed for OMNI
  end subroutine

! IEEE_SUPPORT_ROUNDING

  pure logical function IEEE_SUPPORT_ROUNDING_4 (ROUND_VALUE, X) result(res)
    implicit none
    real(kind=4), intent(in) :: X
    type(IEEE_ROUND_TYPE), intent(in) :: ROUND_VALUE
    ! Only signature needed for OMNI
  end function

  pure logical function IEEE_SUPPORT_ROUNDING_8 (ROUND_VALUE, X) result(res)
    implicit none
    real(kind=8), intent(in) :: X
    type(IEEE_ROUND_TYPE), intent(in) :: ROUND_VALUE
    ! Only signature needed for OMNI
  end function

#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_ROUNDING_10 (ROUND_VALUE, X) result(res)
    implicit none
    real(kind=10), intent(in) :: X
    type(IEEE_ROUND_TYPE), intent(in) :: ROUND_VALUE
    ! Only signature needed for OMNI
  end function
#endif

#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_ROUNDING_16 (ROUND_VALUE, X) result(res)
    implicit none
    real(kind=16), intent(in) :: X
    type(IEEE_ROUND_TYPE), intent(in) :: ROUND_VALUE
    ! Only signature needed for OMNI
  end function
#endif

  pure logical function IEEE_SUPPORT_ROUNDING_NOARG (ROUND_VALUE) result(res)
    implicit none
    type(IEEE_ROUND_TYPE), intent(in) :: ROUND_VALUE
    ! Only signature needed for OMNI
  end function

! IEEE_SUPPORT_UNDERFLOW_CONTROL

  pure logical function IEEE_SUPPORT_UNDERFLOW_CONTROL_4 (X) result(res)
    implicit none
    real(kind=4), intent(in) :: X
    ! Only signature needed for OMNI
  end function

  pure logical function IEEE_SUPPORT_UNDERFLOW_CONTROL_8 (X) result(res)
    implicit none
    real(kind=8), intent(in) :: X
    ! Only signature needed for OMNI
  end function

#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_UNDERFLOW_CONTROL_10 (X) result(res)
    implicit none
    real(kind=10), intent(in) :: X
    ! Only signature needed for OMNI
  end function
#endif

#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_UNDERFLOW_CONTROL_16 (X) result(res)
    implicit none
    real(kind=16), intent(in) :: X
    ! Only signature needed for OMNI
  end function
#endif

  pure logical function IEEE_SUPPORT_UNDERFLOW_CONTROL_NOARG () result(res)
    implicit none
    ! Only signature needed for OMNI
  end function

! IEEE_SUPPORT_* functions

! IEEE_SUPPORT_DATATYPE
  pure logical function IEEE_SUPPORT_DATATYPE_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_DATATYPE_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_DATATYPE_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_DATATYPE_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_DATATYPE_NOARG () result(res)
    implicit none
    res = .true.
  end function

! IEEE_SUPPORT_DENORMAL
  pure logical function IEEE_SUPPORT_DENORMAL_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_DENORMAL_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_DENORMAL_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_DENORMAL_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_DENORMAL_NOARG () result(res)
    implicit none
    res = .true.
  end function

! IEEE_SUPPORT_DIVIDE
  pure logical function IEEE_SUPPORT_DIVIDE_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_DIVIDE_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_DIVIDE_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_DIVIDE_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_DIVIDE_NOARG () result(res)
    implicit none
    res = .true.
  end function


! IEEE_SUPPORT_INF
  pure logical function IEEE_SUPPORT_INF_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_INF_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_INF_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_INF_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_INF_NOARG () result(res)
    implicit none
    res = .true.
  end function

! IEEE_SUPPORT_IO
  pure logical function IEEE_SUPPORT_IO_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_IO_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_IO_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_IO_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_IO_NOARG () result(res)
    implicit none
    res = .true.
  end function

! IEEE_SUPPORT_NAN
  pure logical function IEEE_SUPPORT_NAN_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_NAN_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_NAN_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_NAN_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_NAN_NOARG () result(res)
    implicit none
    res = .true.
  end function

! IEEE_SUPPORT_SQRT
  pure logical function IEEE_SUPPORT_SQRT_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_SQRT_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_SQRT_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_SQRT_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_SQRT_NOARG () result(res)
    implicit none
    res = .true.
  end function

! IEEE_SUPPORT_STANDARD
  pure logical function IEEE_SUPPORT_STANDARD_4 (X) result(res)
    implicit none
    real(4), intent(in) :: X
    res = .true.
  end function
  pure logical function IEEE_SUPPORT_STANDARD_8 (X) result(res)
    implicit none
    real(8), intent(in) :: X
    res = .true.
  end function
#ifdef HAVE_GFC_REAL_10
  pure logical function IEEE_SUPPORT_STANDARD_10 (X) result(res)
    implicit none
    real(10), intent(in) :: X
    res = .true.
  end function
#endif
#ifdef HAVE_GFC_REAL_16
  pure logical function IEEE_SUPPORT_STANDARD_16 (X) result(res)
    implicit none
    real(16), intent(in) :: X
    res = .true.
  end function
#endif
  pure logical function IEEE_SUPPORT_STANDARD_NOARG () result(res)
    implicit none
    res = .true.
  end function

end module IEEE_ARITHMETIC
