# 1 "xmp_constant_f.f90"
# 1 "<built-in>"
# 1 "<command-line>"
# 31 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 32 "<command-line>" 2
# 1 "xmp_constant_f.f90"
!! This file is for making xmp_constant_fmod.f90
!! Run: cpp -E -I$(OMNI_HONE)/include xmp_constant_f.f90 -o xmp_constant_fmod.f90
!!
# 1 "/home/msato/xmp/include/xmp_constant.h" 1
# 5 "xmp_constant_f.f90" 2



integer, parameter :: XMP_BOOL = 502
integer, parameter :: XMP_CHAR = 503
integer, parameter :: XMP_UNSIGNED_CHAR = 504
integer, parameter :: XMP_SHORT = 505
integer, parameter :: XMP_UNSIGNED_SHORT = 506
integer, parameter :: XMP_INT = 507
integer, parameter :: XMP_UNSIGNED_INT = 508
integer, parameter :: XMP_LONG = 509
integer, parameter :: XMP_UNSIGNED_LONG = 510
integer, parameter :: XMP_LONGLONG = 511
integer, parameter :: XMP_UNSIGNED_LONGLONG = 512
integer, parameter :: XMP_FLOAT = 513
integer, parameter :: XMP_DOUBLE = 514
integer, parameter :: XMP_LONG_DOUBLE = 515

integer, parameter :: XMP_SUM = 300
integer, parameter :: XMP_PROD = 301
integer, parameter :: XMP_BAND = 302
integer, parameter :: XMP_LAND = 303
integer, parameter :: XMP_BOR = 304
integer, parameter :: XMP_LOR = 305
integer, parameter :: XMP_BXOR = 306
integer, parameter :: XMP_LXOR = 307
integer, parameter :: XMP_MAX = 308
integer, parameter :: XMP_MIN = 309
integer, parameter :: XMP_FIRSTMAX = 310
integer, parameter :: XMP_FIRSTMIN = 311
integer, parameter :: XMP_LASTMAX = 312
integer, parameter :: XMP_LASTMIN = 313
integer, parameter :: XMP_EQV = 314
integer, parameter :: XMP_NEQV = 315
integer, parameter :: XMP_MINUS = 316
integer, parameter :: XMP_MAXLOC = 317
integer, parameter :: XMP_MINLOC = 318
