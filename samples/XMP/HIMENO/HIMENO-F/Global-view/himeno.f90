! The Himeno Benchmark is made available under the LGPL2.0 or later.
! The original HIMENO Benchmark, is developed by Ryutaro Himeno,
! is available in http://accc.riken.jp/2444.htm
! The XMP version of the HIMENO Benchmarks is developed by Hitoshi Murai and Takenori Shimosaka.
!---------------------------------------------------------------------------
      module alloc1
      include 'xmp_lib.h'
      PARAMETER (mimax = 128, mjmax = 512, mkmax = 512)
      real p(mimax,mjmax,mkmax)
      real a(mimax,mjmax,mkmax,4)
      real b(mimax,mjmax,mkmax,3)
      real c(mimax,mjmax,mkmax,3)
      real bnd(mimax,mjmax,mkmax)
      real wrk1(mimax,mjmax,mkmax), wrk2(mimax,mjmax,mkmax)
      real omega
      integer imax, jmax, kmax
!$xmp nodes n(2,4)
!$xmp template t(mimax,mjmax,mkmax)
!$xmp distribute t(*,block,block) onto n
!$xmp align (*,j,k) with t(*,j,k) :: p, bnd, wrk1, wrk2
!$xmp align (*,j,k,*) with t(*,j,k) :: a, b, c
!$xmp shadow p(0,2:1,2:1)
!$xmp shadow a(0,2:1,2:1,0)
!$xmp shadow b(0,2:1,2:1,0)
!$xmp shadow c(0,2:1,2:1,0)
!$xmp shadow bnd(0,2:1,2:1)
!$xmp shadow wrk1(0,2:1,2:1)
!$xmp shadow wrk2(0,2:1,2:1)
      end module alloc1

      program himeno
      use alloc1

!     ttarget specifys the measuring period in sec
      PARAMETER (ttarget = 60.0)

      integer nn, myrank
      real gosa
      double precision xmp_wtime, cpu0, cpu1, cpu

      myrank = xmp_node_num()

      omega = 0.8
      imax = mimax
      jmax = mjmax
      kmax = mkmax

! Initializing matrixes
      call initmt()

      if (myrank == 1) then
         write(*,*) ' mimax=', mimax, ' mjmax=', mjmax, ' mkmax=', mkmax
         write(*,*) ' imax=', imax, ' jmax=', jmax, ' kmax=', kmax
      end if

! Start measuring

      nn = 3
      if (myrank == 1) then
         write(*,*) ' Start rehearsal measurement process.'
         write(*,*) ' Measure the performance in 3 times.'
      end if

! Jacobi iteration
      cpu0 = xmp_wtime()
      call jacobi(nn, gosa)
      cpu1 = xmp_wtime()

      cpu = cpu1 - cpu0
      flop = real(kmax-2) * real(jmax-2) * real(imax-2) * 34.0 * real(nn)
      xmflops2 = flop / cpu * 1.0e-6

      if (myrank == 1) then
         write(*,*) '  MFLOPS:', xmflops2, '  time(s):', cpu, gosa
      end if

!     end the test loop
      nn = int(ttarget/(cpu/3.0))
!$xmp reduction (max:nn)

      if (myrank == 1) then
         write(*,*) 'Now, start the actual measurement process.'
         write(*,*) 'The loop will be excuted in',nn,' times.'
         write(*,*) 'This will take about one minute.'
         write(*,*) 'Wait for a while.'
      end if

! Jacobi iteration
      cpu0 = xmp_wtime()
      call jacobi(nn, gosa)
      cpu1 = xmp_wtime()

      cpu = cpu1 - cpu0
      flop = real(kmax-2) * real(jmax-2) * real(imax-2) * 34.0 * real(nn)
      xmflops2 = flop * 1.0e-6 / cpu

      if (myrank == 1) then
         write(*,*) ' Loop executed for ', nn, ' times'
         write(*,*) ' Gosa :', gosa
         write(*,*) ' MFLOPS:', xmflops2, '  time(s):', cpu
         score = xmflops2 / 82.84
         write(*,*) ' Score based on Pentium III 600MHz :', score
      end if

      END


!**************************************************************
      subroutine initmt()
!**************************************************************
      use alloc1

!$xmp loop (j,k) on t(*,j,k)
!$omp parallel do
      do k = 1, kmax
         do j = 1, jmax
            do i = 1, imax
               a(i,j,k,1) = 1.0
               a(i,j,k,2) = 1.0
               a(i,j,k,3) = 1.0
               a(i,j,k,4) = 1.0/6.0
               b(i,j,k,1) = 0.0
               b(i,j,k,2) = 0.0
               b(i,j,k,3) = 0.0
               c(i,j,k,1) = 1.0
               c(i,j,k,2) = 1.0
               c(i,j,k,3) = 1.0
               p(i,j,k)  = float(k-1)*float(k-1)/(float(kmax-1)     &
                           *float(kmax-1))
               wrk1(i,j,k) = 0.0
               bnd(i,j,k) = 1.0
            enddo
         enddo
      enddo
!$omp end parallel do
      return
      end

!*************************************************************
      subroutine jacobi(nn,gosa)
!*************************************************************
      use alloc1

!$xmp reflect (p)
      DO loop = 1, nn

         gosa = 0.0
!$xmp loop (J,K) on t(*,J,K)
!$omp parallel do reduction(+:GOSA) private(S0,SS)
         DO K = 2, kmax-1
            DO J = 2, jmax-1
               DO I = 2, imax-1
                  S0 = a(I,J,K,1)*p(I+1,J,K)+a(I,J,K,2)*p(I,J+1,K)  &
                       +a(I,J,K,3)*p(I,J,K+1)                       &
                       +b(I,J,K,1)*(p(I+1,J+1,K)-p(I+1,J-1,K)       &
                       -p(I-1,J+1,K)+p(I-1,J-1,K))                  &
                       +b(I,J,K,2)*(p(I,J+1,K+1)-p(I,J-1,K+1)       &
                       -p(I,J+1,K-1)+p(I,J-1,K-1))                  &
                       +b(I,J,K,3)*(p(I+1,J,K+1)-p(I-1,J,K+1)       &
                       -p(I+1,J,K-1)+p(I-1,J,K-1))                  &
                       +c(I,J,K,1)*p(I-1,J,K)+c(I,J,K,2)*p(I,J-1,K) &
                       +c(I,J,K,3)*p(I,J,K-1)+wrk1(I,J,K)
                  SS = (S0*a(I,J,K,4)-p(I,J,K))*bnd(I,J,K)
                  GOSA = GOSA + SS * SS
                  wrk2(I,J,K) = p(I,J,K)+OMEGA *SS
               enddo
            enddo
         enddo
!$omp end parallel do
         
!$xmp loop (J,K) on t(*,J,K)
!$omp parallel do
         DO K = 2, kmax-1
            DO J = 2, jmax-1
               DO I = 2, imax-1
                  p(I,J,K) = wrk2(I,J,K)
               enddo
            enddo
         enddo
!$omp end parallel do

!$xmp reflect (p)
!$xmp reduction (+:GOSA)
      enddo
! End of iteration
      return
      end
