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
  real(4),dimension(:,:,:),allocatable :: p
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
  integer :: npx(2),npy(2),npz(2)
  integer :: npe,id
  integer :: ijvec,jkvec,ikvec
  integer :: mpi_comm_cart
end module comm
!
program HimenoBMTxp_f90_MPI
!
  use others
  use comm
!
  implicit none
!
  include 'mpif.h'
!
!     ttarget specifys the measuring period in sec
  integer :: mx,my,mz
  integer :: nn,it,ierr
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
     print *
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
  call mpi_barrier(mpi_comm_world,ierr)
  cpu0= mpi_wtime()
!! Jacobi iteration
  call jacobi(nn,gosa)
  cpu1= mpi_wtime() - cpu0
!
  call mpi_allreduce(cpu1, &
                     cpu, &
                     1, &
                     mpi_real8, &
                     mpi_max, &
                     mpi_comm_world, &
                     ierr)
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
  call mpi_barrier(mpi_comm_world,ierr)
  cpu0= mpi_wtime()
!! Jacobi iteration
  call jacobi(nn,gosa)
  cpu1= mpi_wtime() - cpu0
!
  call mpi_allreduce(cpu1, &
                     cpu, &
                     1, &
                     mpi_real8, &
                     mpi_max, &
                     mpi_comm_world, &
                     ierr)
!
  if(id == 0) then
     if(cpu /= 0.0)  xmflops2=flop*1.0d-6/cpu*real(nn)
     print *,' Loop executed for ',nn,' times'
     print *,' Gosa :',gosa
     print *,' MFLOPS:',xmflops2, '  time(s):',cpu
     score=xmflops2/82.84
     print *,' Score based on Pentium III 600MHz :',score
  end if
  call mpi_finalize(ierr)
!
  stop
end program HimenoBMTxp_f90_MPI
!
!
subroutine readparam
!
  use comm
!
  implicit none
!
  include 'mpif.h'
!
  integer :: itmp(3),ierr
  character(10) :: size
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
     print *
     print *,'For example: '
     print *,'DDM pattern= '
     print *,'     1 1 2'
     print *,'     i-direction partitioning : 1'
     print *,'     j-direction partitioning : 1'
     print *,'     k-direction partitioning : 2'
     print *,' DDM pattern = '
     read(*,*) itmp(1),itmp(2),itmp(3)
     print *
  end if
!
  call mpi_bcast(itmp, &
                 3, &
                 mpi_integer, &
                 0, &
                 mpi_comm_world, &
                 ierr)
  call mpi_bcast(size, &
                 10, &
                 mpi_character, &
                 0, &
                 mpi_comm_world, &
                 ierr)
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
  include 'mpif.h'
!
  integer,intent(in) :: nn
  real(4),intent(inout) :: gosa
  integer :: i,j,k,loop,ierr
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
     call sendp(ndx,ndy,ndz)
!
     call mpi_allreduce(wgosa, &
                        gosa, &
                        1, &
                        mpi_real4, &
                        mpi_sum, &
                        mpi_comm_world, &
                        ierr)
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
  use comm
!
  implicit none
!
  include 'mpif.h'
!
  integer :: ierr,icomm,idm(3)
  logical :: ipd(3),ir
!
  call mpi_init(ierr)
  call mpi_comm_size(mpi_comm_world,npe,ierr)
  call mpi_comm_rank(mpi_comm_world,id,ierr)
!
  call readparam
!
  if(ndx*ndy*ndz /= npe) then
     if(id == 0) then
        print *,'Invalid number of PE'
        print *,'Please check partitioning pattern or number of PE'
     end if
     call mpi_finalize(ierr)
     stop
  end if
!
  icomm= mpi_comm_world
!
  idm(1)= ndx
  idm(2)= ndy
  idm(3)= ndz
!
  ipd(1)= .false.
  ipd(2)= .false.
  ipd(3)= .false.
  ir= .false.
!
  call mpi_cart_create(icomm, &
                       ndims, &
                       idm, &
                       ipd, &
                       ir, &
                       mpi_comm_cart, &
                       ierr)
  call mpi_cart_get(mpi_comm_cart, &
                    ndims, &
                    idm, &
                    ipd, &
                    iop, &
                    ierr)
!
  if(ndz > 1) then
     call mpi_cart_shift(mpi_comm_cart, &
                         2, &
                         1, &
                         npz(1), &
                         npz(2), &
                         ierr)
  end if
!
  if(ndy > 1) then
     call mpi_cart_shift(mpi_comm_cart, &
                         1, &
                         1, &
                         npy(1), &
                         npy(2), &
                         ierr)
  end if
!
  if(ndx > 1) then
     call mpi_cart_shift(mpi_comm_cart, &
                         0, &
                         1, &
                         npx(1), &
                         npx(2), &
                         ierr)
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
  include 'mpif.h'
!
  integer,intent(in) :: mx,my,mz
  integer,intent(out) :: ks
  integer :: i,itmp,ierr
  integer :: mx1(0:ndx),my1(0:ndy),mz1(0:ndz)
  integer :: mx2(0:ndx),my2(0:ndy),mz2(0:ndz)
!
!!  define imax, communication direction
  itmp= mx/ndx
  mx1(0)= 0
  do  i=1,ndx
     if(i <= mod(mx,ndx)) then
        mx1(i)= mx1(i-1) + itmp + 1
     else
        mx1(i)= mx1(i-1) + itmp
     end if
  end do
  do i=0,ndx-1
     mx2(i)= mx1(i+1) - mx1(i)
     if(i /= 0)     mx2(i)= mx2(i) + 1
     if(i /= ndx-1) mx2(i)= mx2(i) + 1
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
  imax= mx2(iop(1))
  jmax= my2(iop(2))
  kmax= mz2(iop(3))
!
  mimax= imax + 1
  mjmax= jmax + 1
  mkmax= kmax + 1
!
  if(iop(3) == 0) then
     ks= mz1(iop(3))
  else
     ks= mz1(iop(3)) - 1
  end if
!
!!  j-k vector
  if(ndx > 1) then
     call mpi_type_vector(jmax*kmax, &
                          1, &
                          mimax, &
                          mpi_real4, &
                          jkvec, &
                          ierr)
     call mpi_type_commit(jkvec, &
                          ierr)
  end if
!
!!  i-k vector
  if(ndy > 1) then
     call mpi_type_vector(kmax, &
                          imax, &
                          mimax*mjmax, &
                          mpi_real4, &
                          ikvec, &
                          ierr)
     call mpi_type_commit(ikvec, &
                          ierr)
  end if
!
!!  i-j vector
  if(ndz > 1) then
     call mpi_type_vector(jmax, &
                          imax, &
                          mimax, &
                          mpi_real4, &
                          ijvec, &
                          ierr)
     call mpi_type_commit(ijvec, &
                          ierr)
  end if
!
  return
end subroutine initmax
!
!
!
subroutine sendp(ndx,ndy,ndz)
!
  implicit none
!
  integer,intent(in) :: ndx,ndy,ndz
!
  if(ndz > 1) then
     call sendp3()
  end if
!
  if(ndy > 1) then
     call sendp2()
  end if
!
  if(ndx > 1) then
     call sendp1()
  end if
!
  return
end subroutine sendp
!
!
!
subroutine sendp3()
!
  use pres
  use others
  use comm
!
  implicit none
!
  include 'mpif.h'
!
  integer :: ist(mpi_status_size,0:3),ireq(0:3)=(/-1,-1,-1,-1/)
  integer :: ierr
!
  call mpi_irecv(p(1,1,kmax), &
                 1, &
                 ijvec, &
                 npz(2), &
                 1, &
                 mpi_comm_cart, &
                 ireq(3), &
                 ierr)
!
  call mpi_irecv(p(1,1,1), &
                 1, &
                 ijvec, &
                 npz(1), &
                 2, &
                 mpi_comm_cart, &
                 ireq(2), &
                 ierr)
!
  call mpi_isend(p(1,1,2), &
                 1, &
                 ijvec, &
                 npz(1), &
                 1, &
                 mpi_comm_cart, &
                 ireq(0), &
                 ierr)
!
  call mpi_isend(p(1,1,kmax-1), &
                 1, &
                 ijvec, &
                 npz(2), &
                 2, &
                 mpi_comm_cart, &
                 ireq(1), &
                 ierr)
!
  call mpi_waitall(4, &
                   ireq, &
                   ist, &
                   ierr)
!
  return
end subroutine sendp3
!
!
!
subroutine sendp2()
!
  use pres
  use others
  use comm
!
  implicit none
!
  include 'mpif.h'
!
  integer :: ist(mpi_status_size,0:3),ireq(0:3)=(/-1,-1,-1,-1/)
  integer :: ierr
!
  call mpi_irecv(p(1,1,1), &
                 1, &
                 ikvec, &
                 npy(1), &
                 2, &
                 mpi_comm_cart, &
                 ireq(3), &
                 ierr)
!
  call mpi_irecv(p(1,jmax,1), &
                 1, &
                 ikvec, &
                 npy(2), &
                 1, &
                 mpi_comm_cart, &
                 ireq(2), &
                 ierr)
!
  call mpi_isend(p(1,2,1), &
                 1, &
                 ikvec, &
                 npy(1), &
                 1, &
                 mpi_comm_cart, &
                 ireq(0), &
                 ierr)
!
  call mpi_isend(p(1,jmax-1,1), &
                 1, &
                 ikvec, &
                 npy(2), &
                 2, &
                 mpi_comm_cart, &
                 ireq(1), &
                 ierr)
!
  call mpi_waitall(4, &
                   ireq, &
                   ist, &
                   ierr)
!
  return
end subroutine sendp2
!
!
!
subroutine sendp1()
!
  use pres
  use others
  use comm
!
  implicit none
!
  include 'mpif.h'
!
  integer :: ist(mpi_status_size,0:3),ireq(0:3)=(/-1,-1,-1,-1/)
  integer :: ierr
!
  call mpi_irecv(p(1,1,1), &
                 1, &
                 jkvec, &
                 npx(1), &
                 2, &
                 mpi_comm_cart, &
                 ireq(3), &
                 ierr)
!
  call mpi_irecv(p(imax,1,1), &
                 1, &
                 jkvec, &
                 npx(2), &
                 1, &
                 mpi_comm_cart, &
                 ireq(2), &
                 ierr)
!
  call mpi_isend(p(2,1,1), &
                 1, &
                 jkvec, &
                 npx(1), &
                 1, &
                 mpi_comm_cart, &
                 ireq(0), &
                 ierr)
!
  call mpi_isend(p(imax-1,1,1), &
                 1, &
                 jkvec, &
                 npx(2), &
                 2, &
                 mpi_comm_cart, &
                 ireq(1), &
                 ierr)
!
  call mpi_waitall(4, &
                   ireq, &
                   ist, &
                   ierr)
!
  return
end subroutine sendp1
