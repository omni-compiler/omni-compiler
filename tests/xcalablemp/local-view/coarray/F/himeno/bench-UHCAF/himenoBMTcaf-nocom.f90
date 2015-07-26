!*********************************************************************
!
! This benchmark test program is measuring a cpu performance
! of floating point operation by a Poisson equation solver.
!!
! If you have any question, please ask me via email.
! written by Ryutaro HIMENO, November 26, 2001.
! Version 3.0
! ----------------------------------------------
! Ryutaro Himeno, Dr. of Eng.
! Head of Computer Information Division,
! RIKEN (The Institute of Pysical and Chemical Research)
! Email : himeno@postman.riken.go.jp
! -----------------------------------------------------------
! You can adjust the size of this benchmark code to fit your target
! computer. In that case, please chose following sets of
! (mimax,mjmax,mkmax):
! small : 65,33,33
! small : 129,65,65
! midium: 257,129,129
! large : 513,257,257
! ext.large: 1025,513,513
! This program is to measure a computer performance in MFLOPS
! by using a kernel which appears in a linear solver of pressure
! Poisson eq. which appears in an incompressible Navier-Stokes solver.
! A point-Jacobi method is employed in this solver as this method can 
! be easyly vectrized and be parallelized.
! ------------------
! Finite-difference method, curvilinear coodinate system
! Vectorizable and parallelizable on each grid point
! No. of grid points : imax x jmax x kmax including boundaries
! ------------------
! A,B,C:coefficient matrix, wrk1: source term of Poisson equation
! wrk2 : working area, OMEGA : relaxation parameter
! BND:control variable for boundaries and objects ( = 0 or 1)
! P: pressure
! -------------------
!
module pres
  implicit none
!!!  include 'xmp_coarray.h'
  real(4),dimension(:,:,:),allocatable :: p
  real(4), allocatable, dimension(:,:), codimension[:,:,:] :: &
       buf1l, buf1u,  buf2l, buf2u,  buf3l, buf3u
end module pres
!
module mtrx
  implicit none
  real(4),dimension(:,:,:,:),allocatable :: a,b,c
end module mtrx
!
module bound
  implicit none
  real(4),dimension(:,:,:),allocatable :: bnd
end module bound
!
module work
  implicit none
  real(4),dimension(:,:,:),allocatable :: wrk1,wrk2
end module work
!
module others
  implicit none
  integer :: mx0,my0,mz0
  integer :: mimax,mjmax,mkmax
  integer :: imax,jmax,kmax
  real(4),parameter :: omega=0.8
end module others
!
module comm
  implicit none
  integer,parameter :: ndims=3
  integer :: ndx,ndy,ndz
  integer :: iop(3)
  integer :: npe,id
end module comm

program HimenoBMTxp_f90_CAF
!
  use pres
  use others
  use comm
!
  implicit none
!
  include 'mpif.h'
!
!     ttarget specifys the measuring period in sec
  integer :: mx,my,mz
  integer :: nn,it
  real(4) :: gosa,score
  real(4),parameter :: ttarget=60.0
  real(8) :: cpu,cpu0,cpu1,xmflops2,flop
!
!! Initializing communicator
  call initcomm
!
  mx= mx0-1
  my= my0-1
  mz= mz0-1
!
!! Initializaing computational index
  call initmax(mx,my,mz,it)
!
  call initmem
!
  allocate(buf1l(mjmax, mkmax)[ndx,ndy,*])
  allocate(buf1u(mjmax, mkmax)[ndx,ndy,*])
  allocate(buf2l(mimax, mkmax)[ndx,ndy,*])
  allocate(buf2u(mimax, mkmax)[ndx,ndy,*])
  allocate(buf3l(mimax, mjmax)[ndx,ndy,*])
  allocate(buf3u(mimax, mjmax)[ndx,ndy,*])
!
!! Initializing matrixes
  call initmt(mz,it)
!
  if(id == 0) then
     print *,'Sequential version array size'
     print *,' mimax=',mx0,' mjmax=',my0,' mkmax=',mz0
     print *,'Parallel version  array size'
     print *,' mimax=',mimax,' mjmax=',mjmax,' mkmax=',mkmax
     print *,' imax=',imax,' jmax=',jmax,' kmax=',kmax
     print *,' I-decomp= ',ndx,' J-decomp= ',ndy,' K-decomp= ',ndz
     print *,''                                ! modified for #407
  end if
!
!! Start measuring
!
  nn=3
  if(id == 0) then
     print *,' Start rehearsal measurement process.'
     print *,' Measure the performance in 3 times.'
  end if
!
  gosa= 0.0
  cpu= 0.0
  sync all
  cpu0= mpi_wtime()
!! Jacobi iteration
  call jacobi(nn,gosa)
  cpu1= mpi_wtime() - cpu0
!
!!!  call co_max(cpu1, cpu)
  call co_max(cpu1)
  cpu = cpu1
!
  flop=real(mx-2)*real(my-2)*real(mz-2)*34.0
  if(cpu /= 0.0) xmflops2=flop/cpu*1.0d-6*real(nn)
  if(id == 0) then
     print *,'  MFLOPS:',xmflops2,'  time(s):',cpu,gosa
  end if
  nn= int(ttarget/(cpu/3.0))
!
! end the test loop
  if(id == 0) then
     print *,'Now, start the actual measurement process.'
     print *,'The loop will be excuted in',nn,' times.'
     print *,'This will take about one minute.'
     print *,'Wait for a while.'
  end if
!
  gosa= 0.0
  cpu= 0.0
  sync all
  cpu0= mpi_wtime()
!! Jacobi iteration
  call jacobi(nn,gosa)
  cpu1= mpi_wtime() - cpu0
!
!!!  call co_max(cpu1, cpu)
  call co_max(cpu1)
  cpu = cpu1
!
  if(id == 0) then
     if(cpu /= 0.0)  xmflops2=flop*1.0d-6/cpu*real(nn)
     print *,' Loop executed for ',nn,' times'
     print *,' Gosa :',gosa
     print *,' MFLOPS:',xmflops2, '  time(s):',cpu
     score=xmflops2/82.84
     print *,' Score based on Pentium III 600MHz :',score
  end if
!
  stop
end program HimenoBMTxp_f90_CAF
!
!
subroutine readparam
!
  use pres
  use comm
!
  implicit none
!
  integer, save :: itmp(3)[*]
  character(12), save :: size[*]
!
  if(id == 0) then
     print *,'For example:'
     print *,'Grid-size= '
     print *,'           XS  (64x32x32)'
     print *,'           S   (128x64x64)'
     print *,'           M   (256x128x128)'
     print *,'           L   (512x256x256)'
     print *,'           XL  (1024x512x512)'
     print *,' Grid-size = '
     read(*,*) size
     print *,''                                ! modified for #407
     print *,'For example: '
     print *,'DDM pattern= '
     print *,'     1 1 2'
     print *,'     i-direction partitioning : 1'
     print *,'     j-direction partitioning : 1'
     print *,'     k-direction partitioning : 2'
     print *,' DDM pattern = '
     read(*,*) itmp(1),itmp(2),itmp(3)
     print *,''                                ! modified for #407
  end if
!
  sync all
  if (id /= 0) then
     itmp(:) = itmp(:)[1]
     size = size[1]
  end if
  sync all
!
  ndx= itmp(1)
  ndy= itmp(2)
  ndz= itmp(3)
!
  call grid_set(size)
!
  return
end subroutine readparam
!
!
subroutine grid_set(size)
!
  use others
!
  implicit none
!
  character(10),intent(in) :: size
!
  select case(size)
  case("xs")
     mx0=65
     my0=33
     mz0=33
  case("XS")
     mx0=65
     my0=33
     mz0=33
  case("s")
     mx0=129
     my0=65
     mz0=65
  case("S")
     mx0=129
     my0=65
     mz0=65
  case("m")
     mx0=257
     my0=129
     mz0=129
  case("M")
     mx0=257
     my0=129
     mz0=129
  case("l")
     mx0=513
     my0=257
     mz0=257
  case("L")
     mx0=513
     my0=257
     mz0=257
  case("xl")
     mx0=1025
     my0=513
     mz0=513
  case("XL")
     mx0=1025
     my0=513
     mz0=513
  case default
     print *,'Invalid input character !!'
     stop
  end select
!
  return
end subroutine grid_set
!
!
!**************************************************************
subroutine initmt(mz,it)
!**************************************************************
  use pres
  use mtrx
  use bound
  use work
  use others
!
  implicit none
!
  integer,intent(in) :: mz,it
  integer :: i,j,k
!
  a=0.0
  b=0.0
  c=0.0
  p=0.0
  wrk1=0.0   
  wrk2=0.0   
  bnd=0.0 
!
  a(1:imax,1:jmax,1:kmax,1:3)=1.0
  a(1:imax,1:jmax,1:kmax,4)=1.0/6.0
  c(1:imax,1:jmax,1:kmax,:)=1.0
  bnd(1:imax,1:jmax,1:kmax)=1.0 
  do k=1,kmax
     p(:,:,k) =real((k-1+it)*(k-1+it),4)/real((mz-1)*(mz-1),4)
  enddo

!
  return
end subroutine initmt
!
!*************************************************************
subroutine initmem
!*************************************************************
  use pres
  use mtrx
  use bound
  use work
  use others
!
  implicit none
!
  allocate(p(mimax,mjmax,mkmax))
  allocate(a(mimax,mjmax,mkmax,4),b(mimax,mjmax,mkmax,3), &
           c(mimax,mjmax,mkmax,3))
  allocate(bnd(mimax,mjmax,mkmax))
  allocate(wrk1(mimax,mjmax,mkmax),wrk2(mimax,mjmax,mkmax))
!
  return
end subroutine initmem
!
!*************************************************************
subroutine jacobi(nn,gosa)
!*************************************************************
  use pres
  use mtrx
  use bound
  use work
  use comm
  use others
!
  implicit none
!
  integer,intent(in) :: nn
  real(4),intent(inout) :: gosa
  integer :: i,j,k,loop
  real(4) :: s0,ss,wgosa
!  
  do loop=1,nn
     gosa=0.0
     wgosa=0.0
     do k=2,kmax-1
        do j=2,jmax-1
           do i=2,imax-1
              s0=a(I,J,K,1)*p(I+1,J,K) &
                   +a(I,J,K,2)*p(I,J+1,K) &
                   +a(I,J,K,3)*p(I,J,K+1) &
                   +b(I,J,K,1)*(p(I+1,J+1,K)-p(I+1,J-1,K) &
                               -p(I-1,J+1,K)+p(I-1,J-1,K)) &
                   +b(I,J,K,2)*(p(I,J+1,K+1)-p(I,J-1,K+1) &
                               -p(I,J+1,K-1)+p(I,J-1,K-1)) &
                   +b(I,J,K,3)*(p(I+1,J,K+1)-p(I-1,J,K+1) &
                               -p(I+1,J,K-1)+p(I-1,J,K-1)) &
                   +c(I,J,K,1)*p(I-1,J,K) &
                   +c(I,J,K,2)*p(I,J-1,K) &
                   +c(I,J,K,3)*p(I,J,K-1)+wrk1(I,J,K)
              ss=(s0*a(I,J,K,4)-p(I,J,K))*bnd(I,J,K)
              wgosa=wgosa+ss*ss
              wrk2(I,J,K)=p(I,J,K)+omega*ss
           enddo
        enddo
     enddo
!     
     p(2:imax-1,2:jmax-1,2:kmax-1)= &
          wrk2(2:imax-1,2:jmax-1,2:kmax-1)
!
     call sendp()
!
!!!     call co_sum(wgosa, gosa)
     call co_sum(wgosa)
     gosa = wgosa
!
  enddo
!! End of iteration
  return
end subroutine jacobi
!
!
!
subroutine initcomm
!
  use pres
  use comm
  use others
!
  implicit none
!
  integer,allocatable:: dummy(:)[:,:,:]
!
  npe = num_images()
  id = this_image() - 1
!
  call readparam
!
  allocate(dummy(0)[ndx,ndy,*])
  iop = this_image(dummy) - 1
!
  if(ndx*ndy*ndz /= npe) then
     if(id == 0) then
        print *,'Invalid number of PE'
        print *,'Please check partitioning pattern or number of PE'
     end if
     stop
  end if
!
  return
end subroutine initcomm
!
!
!
subroutine initmax(mx,my,mz,ks)
  use others
  use comm
!
  implicit none
!
  integer,intent(in) :: mx,my,mz
  integer,intent(out) :: ks
  integer :: i,itmp
  integer :: mx1(0:ndx),my1(0:ndy),mz1(0:ndz)
  integer :: mx2(0:ndx),my2(0:ndy),mz2(0:ndz)
!
!!  define imax, communication direction
  itmp= mx/ndx
  mx1(0)= 0
  do  i=1,ndx
     if(i <= mod(mx,ndx)) then
        mx1(i)= mx1(i-1) + itmp + 1    !! get my global lbound
     else
        mx1(i)= mx1(i-1) + itmp        !! get my global lbound
     end if
  end do
  do i=0,ndx-1
     mx2(i)= mx1(i+1) - mx1(i)             !! get my width
     if(i /= 0)     mx2(i)= mx2(i) + 1     !! add lower shadow
     if(i /= ndx-1) mx2(i)= mx2(i) + 1     !! add upper shadow
  end do
!
  itmp= my/ndy
  my1(0)= 0
  do  i=1,ndy
     if(i <= mod(my,ndy)) then
        my1(i)= my1(i-1) + itmp + 1
     else
        my1(i)= my1(i-1) + itmp
     end if
  end do
  do i=0,ndy-1
     my2(i)= my1(i+1) - my1(i)
     if(i /= 0)      my2(i)= my2(i) + 1
     if(i /= ndy-1)  my2(i)= my2(i) + 1
  end do
!
  itmp= mz/ndz
  mz1(0)= 0
  do  i=1,ndz
     if(i <= mod(mz,ndz)) then
        mz1(i)= mz1(i-1) + itmp + 1
     else
        mz1(i)= mz1(i-1) + itmp
     end if
  end do
  do i=0,ndz-1
     mz2(i)= mz1(i+1) - mz1(i)
     if(i /= 0)      mz2(i)= mz2(i) + 1
     if(i /= ndz-1)  mz2(i)= mz2(i) + 1
  end do
!
  mimax=maxval(mx2(0:ndx-1)) + 1
  mjmax=maxval(my2(0:ndy-1)) + 1
  mkmax=maxval(mz2(0:ndz-1)) + 1
!
  imax= mx2(iop(1))
  jmax= my2(iop(2))
  kmax= mz2(iop(3))
!
  if(iop(3) == 0) then
     ks= mz1(iop(3))
  else
     ks= mz1(iop(3)) - 1
  end if
!
  return
end subroutine initmax
!
!
!
subroutine sendp()
  return
end subroutine sendp
