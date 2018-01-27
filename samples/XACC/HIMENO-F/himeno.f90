! The Himeno Benchmark is made available under the LGPL2.0 or later.
! The original HIMENO Benchmark, is developed by Ryutaro Himeno,
! is available in http://accc.riken.jp/2444.htm
! The XMP version of the HIMENO Benchmarks is developed by Hitoshi Murai and Takenori Shimosaka.
!---------------------------------------------------------------------------
      module alloc1
      include 'xmp_lib.h'
      PARAMETER (mimax = 512, mjmax = 256, mkmax = 256)
      real p(mimax,mjmax,mkmax)
      real a(mimax,mjmax,mkmax,4)
      real b(mimax,mjmax,mkmax,3)
      real c(mimax,mjmax,mkmax,3)
      real bnd(mimax,mjmax,mkmax)
      real wrk1(mimax,mjmax,mkmax), wrk2(mimax,mjmax,mkmax)
      real omega
      integer imax, jmax, kmax
!$xmp nodes n(1,NDY,NDX)
!$xmp template t(mimax,mjmax,mkmax)
!$xmp distribute t(block,block,block) onto n
!$xmp align (i,j,k) with t(i,j,k) :: p, bnd, wrk1, wrk2
!$xmp align (i,j,k,*) with t(i,j,k) :: a, b, c
!$xmp shadow (0:1,1:2,1:2) :: p, bnd, wrk1, wrk2
!$xmp shadow (0:1,1:2,1:2,0) :: a, b, c
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
         write(0,*) ' mimax=', mimax, ' mjmax=', mjmax, ' mkmax=', mkmax
         write(0,*) ' imax=', imax, ' jmax=', jmax, ' kmax=', kmax
      end if

! Start measuring

      nn = 3
      if (myrank == 1) then
         write(0,*) ' Start rehearsal measurement process.'
         write(0,*) ' Measure the performance in 3 times.'
      end if

!$acc data copyin(p, bnd, wrk1, wrk2, a, b, c)

! Jacobi iteration
      cpu0 = xmp_wtime()
      call jacobi(nn, gosa)
      cpu1 = xmp_wtime()

      cpu = cpu1 - cpu0
      flop = real(kmax-2) * real(jmax-2) * real(imax-2) * 34.0 * real(nn)
      xmflops2 = flop / cpu * 1.0e-6

      if (myrank == 1) then
         write(0,*) '  MFLOPS:', xmflops2, '  time(s):', cpu, gosa
      end if

!     end the test loop
      nn = 1000 !int(ttarget/(cpu/3.0))
!$xmp reduction (max:nn)

      if (myrank == 1) then
         write(0,*) 'Now, start the actual measurement process.'
         write(0,*) 'The loop will be excuted in',nn,' times.'
         write(0,*) 'This will take about one minute.'
         write(0,*) 'Wait for a while.'
      end if

! Jacobi iteration
      cpu0 = xmp_wtime()
      call jacobi(nn, gosa)
      cpu1 = xmp_wtime()

      cpu = cpu1 - cpu0
      flop = real(kmax-2) * real(jmax-2) * real(imax-2) * 34.0 * real(nn)
      xmflops2 = flop * 1.0e-6 / cpu

!$acc end data

      if (myrank == 1) then
         write(0,*) ' Loop executed for ', nn, ' times'
         write(0,*) ' Gosa :', gosa
         write(0,*) ' MFLOPS:', xmflops2, '  time(s):', cpu
         score = xmflops2 / 82.84
         write(0,*) ' Score based on Pentium III 600MHz :', score
         write(*,*) xmflops2, ',', cpu
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

!$acc data present(p, bnd, wrk1, wrk2, a, b, c) create(gosa)
!$xmp reflect (p) width(0,1,1) acc
      DO loop = 1, nn

         gosa = 0.0
!$acc update device(gosa)
!$xmp loop (J,K) on t(*,J,K)
!$acc parallel loop firstprivate(omega, imax, jmax, kmax) reduction(+:gosa) collapse(2) gang vector_length(64) async
         DO K = 2, kmax-1
            DO J = 2, jmax-1
!$acc loop vector reduction(+:gosa) private(s0, ss)
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
         
!$xmp loop (J,K) on t(*,J,K)
!$acc parallel loop firstprivate(imax, jmax, kmax) collapse(2) gang vector_length(64) async
         DO K = 2, kmax-1
            DO J = 2, jmax-1
!$acc loop vector
               DO I = 2, imax-1
                  p(I,J,K) = wrk2(I,J,K)
               enddo
            enddo
         enddo

!$acc wait

!$xmp reflect (p) width(0,1,1) acc
!$acc update host(gosa)
!$xmp reduction (+:GOSA)
      enddo
!$acc end data
! End of iteration
      return
      end
