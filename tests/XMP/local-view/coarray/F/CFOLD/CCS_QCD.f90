  implicit none

  integer, parameter :: NTX=32        !! _NTX
  integer, parameter :: NTY=32        !! _NTY
  integer, parameter :: NTZ=32        !! _NTZ
  integer, parameter :: NTT=32        !! _NTT

!*******************************
! node array size
!*******************************
  integer, parameter :: NDIMX=2       !! _NDIMX
  integer, parameter :: NDIMY=2       !! _NDIMY
  integer, parameter :: NDIMZ=2       !! _NDIMZ

!*******************************
! lattice size/node
!*******************************
  integer, parameter :: NX=NTX/NDIMX
  integer, parameter :: NY=NTY/NDIMY
  integer, parameter :: NZ=NTZ/NDIMZ
  integer, parameter :: NT=NTT

  integer, parameter :: NX1=NX+1
  integer, parameter :: NY1=NY+1
  integer, parameter :: NZ1=NZ+1
  integer, parameter :: NT1=NT+1

  integer, parameter :: NTH=NT/2
  integer, parameter :: NPU=NDIMX*NDIMY*NDIMZ
  integer :: NDIM
  parameter  (NDIM=4)
  integer :: COL
  parameter  (COL=3)
  integer :: SPIN
  parameter  (SPIN=2**(NDIM/2))
  integer :: CLSP
  parameter  (CLSP=COL*SPIN)
  integer :: CLSPH
  parameter  (CLSPH=(CLSP/2+1)*(CLSP/4))
  integer :: FBUFF_MAX_SIZE
  parameter (FBUFF_MAX_SIZE=MAX(CLSP*(NTH+1)*NZ*NY,   &  ! for x-direction w/o cornar
 &                              CLSP*(NTH+1)*NZ*NX,   &  ! for y-direction w/o cornar
 &                              CLSP*(NTH+1)*NY*NX,   &  ! for z-direction w/o cornar
 &                              CLSP*(NTH+1)*NZ*2,    &  ! for x-direction at cornar (x-y plane)
 &                              CLSP*(NTH+1)*NY*2,    &  ! for x-direction at cornar (x-z plane)
 &                              CLSP*(NTH+1)*NX*2  ))    ! for y-direction at cornar (y-z plane)

  real static_coarray(COL)[*]       ! OK
!  real static_coarray(SPIN)[*]
!  real static_coarray(CLSP)[*]
!  real static_coarray(FBUFF_MAX_SIZE)[*]

  write(*,*) "COL,SPIN,CLSP,FBUFF_MAX_SIZE=",COL,SPIN,CLSP,FBUFF_MAX_SIZE

  end program
