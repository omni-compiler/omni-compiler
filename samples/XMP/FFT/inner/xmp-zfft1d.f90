!
!     FFTE: A FAST FOURIER TRANSFORM PACKAGE
!
!     (C) COPYRIGHT SOFTWARE, 2000-2004, 2008-2011, ALL RIGHTS RESERVED
!                BY
!         DAISUKE TAKAHASHI
!         GRADUATE SCHOOL OF SYSTEMS AND INFORMATION ENGINEERING
!         UNIVERSITY OF TSUKUBA
!         1-1-1 TENNODAI, TSUKUBA, IBARAKI 305-8573, JAPAN
!         E-MAIL: daisuke@cs.tsukuba.ac.jp
!
!
!     PARALLEL 1-D COMPLEX FFT ROUTINE
!     FORTRAN77 + MPI SOURCE PROGRAM
!     WRITTEN BY DAISUKE TAKAHASHI
!     rewritten for XMP by T. Shimosaka, M. Sato, and H. Iwashita

      subroutine xmpzfft1d(a, b, c, w, n, nx, ny, c_size, is_back)
      use common
      implicit none
      complex*16 a(nx,ny), b(ny,nx), w(ny,nx)
      complex*16 c(*)
      integer*8 :: n
      integer :: nx, ny, c_size, i, j
      real*8 :: dn
      logical :: is_back

!$xmp template tx(nx)
!$xmp template ty(ny)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,j) with ty(j)
!$xmp align b(*,i) with tx(i)
!$xmp align w(*,i) with tx(i)

      if (is_back) then
!$xmp    loop on ty(j)
!$omp    parallel do
         do j = 1, ny
            do i = 1, nx
               a(i,j) = dconjg(a(i,j))
            end do
         end do
      end if

      if (c_size /= 0) then
         call xmpzfft1d0( a, b, c, c, w, nx, ny )
      else
         call xmpzfft1d0( a, b, b, a, w, nx, ny )
      end if

      dn = 1.0D0 / dble(n)
      if (is_back) then
         !$xmp loop on tx(i)
         !$omp parallel do
         do i = 1, nx
            do j = 1, ny
               b(j,i) = dn * dconjg(b(j,i))
            end do
         end do
      end if

      return
      end subroutine xmpzfft1d

      subroutine xmpzfft1d0(a,b,cx,cy,w,nx,ny)
      use common
      implicit real*8 (a-h,o-z)
!
      complex*16 a(nx,ny), b(ny,nx), w(ny,nx)
      complex*16 cx(*),cy(*)
      integer,external :: omp_get_thread_num

!$xmp template tx(nx)
!$xmp template ty(ny)
!$xmp distribute tx(block) onto p
!$xmp distribute ty(block) onto p
!$xmp align a(*,i) with ty(i)
!$xmp align b(*,i) with tx(i)
!$xmp align w(*,i) with tx(i)

      call xmp_transpose(b,a,1)

      call zfft1d(b,ny,0,cy)  ! init table

!$xmp loop on tx(i)
      do i=1,nx
         call zfft1d(b(1,i),ny,-1,cy)
      end do

!$xmp loop on tx(i)
!$omp parallel do
      do i=1,nx
          do j=1,ny
            b(j,i)=b(j,i)*w(j,i)
          end do
      end do

      call xmp_transpose(a,b,1)

      call zfft1d(a,nx,0,cx)   ! setup

!$xmp loop on ty(j)
      do j=1,ny
        call zfft1d(a(1,j),nx,-1,cx)
      end do

      call xmp_transpose(b,a,1)

      return
      end subroutine xmpzfft1d0

      subroutine xmpsettbl(w,nx,ny)
      use common
      implicit real*8 (a-h,o-z)
      complex*16 w(ny,nx)
!$xmp template t(nx)
!$xmp distribute t(block) onto p
!$xmp align w(*,i) with t(i)

      pi2=8.0d0*datan(1.0d0)
      px=-pi2/(dble(nx)*dble(ny))
!$xmp loop on t(i)
!$omp parallel do private(temp)
      do i=1,nx
        do j=1,ny
          temp=px*(dble(j-1)*dble(i-1))
          w(j,i)=dcmplx(dcos(temp),dsin(temp))
        end do
      end do
      return
      end subroutine xmpsettbl
