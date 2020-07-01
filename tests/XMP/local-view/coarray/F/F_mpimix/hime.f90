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
  integer :: npe,id
!!  include 'xmp_coarray.h'
  real, allocatable, dimension(:,:), codimension[:,:,:] :: &
       buf1l, buf1u,  buf2l, buf2u,  buf3l, buf3u
end module comm

program HimenoBMTxp_f90_CAF
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
  allocate(buf1l(mjmax, mkmax)[ndx,ndy,*])
  allocate(buf1u(mjmax, mkmax)[ndx,ndy,*])
  allocate(buf2l(mimax, mkmax)[ndx,ndy,*])
  allocate(buf2u(mimax, mkmax)[ndx,ndy,*])
  allocate(buf3l(mimax, mjmax)[ndx,ndy,*])
  allocate(buf3u(mimax, mjmax)[ndx,ndy,*])

  iop = this_image(buf1l) - 1
!
!! Initializing matrixes
  call initmt(mz,it)
!
  nn=1
!
  gosa= 0.0
  cpu= 0.0
  sync all
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
  stop
end program HimenoBMTxp_f90_CAF
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
  integer :: itmp(3)[*]
!!!  character(10) :: size[*]     !! to avoid bug #354
  character(12) :: size(1)[*]
!

  size="M"
  itmp=(/2,2,2/)

  sync all
  if (id /= 0) then
     itmp = itmp[1]
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
     mx0=257
     my0=129
     mz0=129
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
     call sendp()
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
  use others
!
  implicit none
!
  include 'mpif.h'
!
  npe = num_images()
  id = this_image() - 1
!
  call readparam
!
!
  if(ndx*ndy*ndz /= npe) then
     if(id == 0) then
        print *,'Invalid number of PE'
        print *,'Please check partitioning pattern or number of PE'
     end if
     stop 1
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
  imax= mx2(iop(1))
  jmax= my2(iop(2))
  kmax= mz2(iop(3))
!
!!!!! debug point #1
  call mpi_allreduce(imax, mimax, 1, mpi_integer, &
                     mpi_max, mpi_comm_world, ierr)
  call mpi_allreduce(jmax, mjmax, 1, mpi_integer, &
                     mpi_max, mpi_comm_world, ierr)
  call mpi_allreduce(kmax, mkmax, 1, mpi_integer, &
                     mpi_max, mpi_comm_world, ierr)
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
!
  use pres
  use others
  use comm
  implicit none
  integer mex, mey, mez

  mex = iop(1) + 1
  mey = iop(2) + 1
  mez = iop(3) + 1

  sync all

  !*** put z-axis
  if (mez>1) then
     buf3u(:,:)[mex,mey,mez-1] = p(:,:,2     )
  end if
  if (mez<ndz) then
     buf3l(:,:)[mex,mey,mez+1] = p(:,:,kmax-1)
  endif

  sync all

  !*** unpack z-axis
  if (mez<ndz) then
     p(:,:,kmax) = buf3u(:,:)
  end if
  if (mez>1) then
     p(:,:,1   ) = buf3l(:,:)
  endif

  sync all

  !*** put y-axis
  if (mey>1) then
     buf2u(:,:)[mex,mey-1,mez] = p(:,2     ,:)
  end if
  if (mey<ndy) then
     buf2l(:,:)[mex,mey+1,mez] = p(:,jmax-1,:)
  endif

  sync all

  !*** unpack y-axis
  if (mey<ndy) then
     p(:,jmax,:) = buf2u(:,:)
  end if
  if (mey>1) then
     p(:,1   ,:) = buf2l(:,:)
  endif

  sync all

  !*** put x-axis
  if (mex>1) then
     buf1u(:,:)[mex-1,mey,mez] = p(2     ,:,:)
  end if
  if (mex<ndx) then
     buf1l(:,:)[mex+1,mey,mez] = p(imax-1,:,:)
  endif

  sync all

  !*** unpack x-axis
  if (mex<ndx) then
     p(imax,:,:) = buf1u(:,:)
  end if
  if (mex>1) then
     p(1   ,:,:) = buf1l(:,:)
  endif

  sync all

  return
end subroutine sendp
