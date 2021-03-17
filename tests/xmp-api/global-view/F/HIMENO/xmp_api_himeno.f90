! The Himeno Benchmark is made available under the LGPL2.0 or later.
! The original HIMENO Benchmark, is developed by Ryutaro Himeno,
! is available in http://accc.riken.jp/2444.htm
! The XMP version of the HIMENO Benchmarks is developed by Hitoshi Murai and Takenori Shimosaka.
!---------------------------------------------------------------------------
      module alloc1
      !include 'xmp_lib.h'
      !--- 2020 Fujitsu
      use xmp_api
      !--- 2020 Fujitsu end
      PARAMETER (mimax = 128, mjmax = 512, mkmax = 512)
      !--- 2020 Fujitsu
      !real p(mimax,mjmax,mkmax)
      !real a(mimax,mjmax,mkmax,4)
      !real b(mimax,mjmax,mkmax,3)
      !real c(mimax,mjmax,mkmax,3)
      !real bnd(mimax,mjmax,mkmax)
      !real wrk1(mimax,mjmax,mkmax), wrk2(mimax,mjmax,mkmax)
      real, allocatable :: p (:, :, :)
      real, allocatable :: a (:, :, :, :)
      real, allocatable :: b (:, :, :, :)
      real, allocatable :: c (:, :, :, :)
      real, allocatable :: bnd (:, :, :)
      real, allocatable :: wrk1 (:, :, :)
      real, allocatable :: wrk2 (:, :, :)
      integer(8) :: p_desc, a_desc, b_desc, c_desc, bnd_desc, wrk1_desc, wrk2_desc
      integer(4) :: j_start, j_end, j_step, k_start, k_end, k_step
      integer(8) :: g_j, g_k
      !--- 2020 Fujitsu end
      real omega
      integer imax, jmax, kmax
      !--- 2020 Fujitsu
! !$xmp nodes n(2,4)
! !$xmp template t(mimax,mjmax,mkmax)
! !$xmp distribute t(*,block,block) onto n
! !$xmp align (*,j,k) with t(*,j,k) :: p, bnd, wrk1, wrk2
! !$xmp align (*,j,k,*) with t(*,j,k) :: a, b, c
! !$xmp shadow p(0,2:1,2:1)
! !$xmp shadow a(0,2:1,2:1,0)
! !$xmp shadow b(0,2:1,2:1,0)
! !$xmp shadow c(0,2:1,2:1,0)
! !$xmp shadow bnd(0,2:1,2:1)
! !$xmp shadow wrk1(0,2:1,2:1)
! !$xmp shadow wrk2(0,2:1,2:1)
      integer(8) :: n_desc, t_desc
      !--- 2020 Fujitsu end
      end module alloc1

      program himeno
      use alloc1

!     ttarget specifys the measuring period in sec
      PARAMETER (ttarget = 60.0)

      integer nn, myrank
      real gosa
      !--- 2020 Fujitsu
      !double precision xmp_wtime, cpu0, cpu1, cpu
      double precision cpu0, cpu1, cpu
      !--- 2020 Fujitsu end

      !--- 2020 Fujitsu
      integer(4), dimension(2) :: node_dims
      integer(8), dimension(4) :: dim_lb, dim_ub
      integer(4), dimension(4) :: local_lb, local_ub
      integer(4) :: status

      call xmp_api_init
      
      !--- !$xmp nodes n(2,4)
      node_dims(1) = 2
      node_dims(2) = 4
      call xmp_global_nodes(n_desc, 2, node_dims, .true.)

      !--- !$xmp template t(mimax,mjmax,mkmax)
      dim_lb(1) = 1; dim_ub(1) = mimax
      dim_lb(2) = 1; dim_ub(2) = mjmax
      dim_lb(3) = 1; dim_ub(3) = mkmax
      call xmp_new_template(t_desc, n_desc, 3, dim_lb, dim_ub)

      !--- !$xmp distribute t(*,block,block) onto n
      call xmp_dist_template_block(t_desc, 2, 2, status)
      call xmp_dist_template_block(t_desc, 3, 3, status)

      !--- !$xmp align (*,j,k) with t(*,j,k) :: p, bnd, wrk1, wrk2
      call xmp_new_array(p_desc, t_desc, XMP_FLOAT, 3, dim_lb, dim_ub)
      call xmp_align_array(p_desc, 2, 2, 0, status)
      call xmp_align_array(p_desc, 3, 3, 0, status)
      call xmp_get_array_local_dim(p_desc, local_lb, local_ub, status)
      allocate(p(local_lb(1):local_ub(1), local_lb(2):local_ub(2), local_lb(3):local_ub(3)))
      call xmp_allocate_array(p_desc, loc(p), status)

      call xmp_new_array(bnd_desc, t_desc, XMP_FLOAT, 3, dim_lb, dim_ub)
      call xmp_align_array(bnd_desc, 2, 2, 0, status)
      call xmp_align_array(bnd_desc, 3, 3, 0, status)
      call xmp_get_array_local_dim(bnd_desc, local_lb, local_ub, status)
      allocate(bnd(local_lb(1):local_ub(1), local_lb(2):local_ub(2), local_lb(3):local_ub(3)))
      call xmp_allocate_array(bnd_desc, loc(bnd), status)

      call xmp_new_array(wrk1_desc, t_desc, XMP_FLOAT, 3, dim_lb, dim_ub)
      call xmp_align_array(wrk1_desc, 2, 2, 0, status)
      call xmp_align_array(wrk1_desc, 3, 3, 0, status)
      call xmp_get_array_local_dim(wrk1_desc, local_lb, local_ub, status)
      allocate(wrk1(local_lb(1):local_ub(1), local_lb(2):local_ub(2), local_lb(3):local_ub(3)))
      call xmp_allocate_array(wrk1_desc, loc(wrk1), status)

      call xmp_new_array(wrk2_desc, t_desc, XMP_FLOAT, 3, dim_lb, dim_ub)
      call xmp_align_array(wrk2_desc, 2, 2, 0, status)
      call xmp_align_array(wrk2_desc, 3, 3, 0, status)
      call xmp_get_array_local_dim(wrk2_desc, local_lb, local_ub, status)
      allocate(wrk2(local_lb(1):local_ub(1), local_lb(2):local_ub(2), local_lb(3):local_ub(3)))
      call xmp_allocate_array(wrk2_desc, loc(wrk2), status)

      !--- !$xmp align (*,j,k,*) with t(*,j,k) :: a, b, c
      !real a(mimax,mjmax,mkmax,4)
      dim_lb(4) = 1; dim_ub(4) = 4
      call xmp_new_array(a_desc, t_desc, XMP_FLOAT, 4, dim_lb, dim_ub)
      call xmp_align_array(a_desc, 2, 2, 0, status)
      call xmp_align_array(a_desc, 3, 3, 0, status)
      call xmp_get_array_local_dim(a_desc, local_lb, local_ub, status)
      allocate(a(local_lb(1):local_ub(1), local_lb(2):local_ub(2), local_lb(3):local_ub(3), local_lb(4):local_ub(4)))
      call xmp_allocate_array(a_desc, loc(a), status)

      !real b(mimax,mjmax,mkmax,3)
      dim_lb(4) = 1; dim_ub(4) = 3
      call xmp_new_array(b_desc, t_desc, XMP_FLOAT, 4, dim_lb, dim_ub)
      call xmp_align_array(b_desc, 2, 2, 0, status)
      call xmp_align_array(b_desc, 3, 3, 0, status)
      call xmp_get_array_local_dim(b_desc, local_lb, local_ub, status)
      allocate(b(local_lb(1):local_ub(1), local_lb(2):local_ub(2), local_lb(3):local_ub(3), local_lb(4):local_ub(4)))
      call xmp_allocate_array(b_desc, loc(b), status)

      !real c(mimax,mjmax,mkmax,3)
      call xmp_new_array(c_desc, t_desc, XMP_FLOAT, 4, dim_lb, dim_ub)
      call xmp_align_array(c_desc, 2, 2, 0, status)
      call xmp_align_array(c_desc, 3, 3, 0, status)
      call xmp_get_array_local_dim(c_desc, local_lb, local_ub, status)
      allocate(c(local_lb(1):local_ub(1), local_lb(2):local_ub(2), local_lb(3):local_ub(3), local_lb(4):local_ub(4)))
      call xmp_allocate_array(c_desc, loc(c), status)

      ! !$xmp shadow p(0,2:1,2:1)
      call xmp_set_shadow(p_desc, 2, 2, 1, status)
      call xmp_set_shadow(p_desc, 3, 2, 1, status)

      ! !$xmp shadow a(0,2:1,2:1,0)
      call xmp_set_shadow(a_desc, 2, 2, 1, status)
      call xmp_set_shadow(a_desc, 3, 2, 1, status)

      ! !$xmp shadow b(0,2:1,2:1,0)
      call xmp_set_shadow(b_desc, 2, 2, 1, status)
      call xmp_set_shadow(b_desc, 3, 2, 1, status)
      
      ! !$xmp shadow c(0,2:1,2:1,0)
      call xmp_set_shadow(c_desc, 2, 2, 1, status)
      call xmp_set_shadow(c_desc, 3, 2, 1, status)

      ! !$xmp shadow bnd(0,2:1,2:1)
      call xmp_set_shadow(bnd_desc, 2, 2, 1, status)
      call xmp_set_shadow(bnd_desc, 3, 2, 1, status)

      ! !$xmp shadow wrk1(0,2:1,2:1)
      call xmp_set_shadow(wrk1_desc, 2, 2, 1, status)
      call xmp_set_shadow(wrk1_desc, 3, 2, 1, status)

      ! !$xmp shadow wrk2(0,2:1,2:1)
      call xmp_set_shadow(wrk2_desc, 2, 2, 1, status)
      call xmp_set_shadow(wrk2_desc, 3, 2, 1, status)
      !--- 2020 Fujitsu end

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
      !--- 2020 Fujitsu
! !$xmp reduction (max:nn)
      call xmp_reduction_scalar(XMP_MAX, XMP_INT, loc(nn), status)
      !--- 2020 Fujitsu end

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

      !--- 2020 Fujitsu
      integer(4) :: status

! !$xmp loop (j,k) on t(*,j,k)
      call xmp_loop_schedule(1, jmax, 1, t_desc, 2, j_start, j_end, j_step, status)
      call xmp_loop_schedule(1, kmax, 1, t_desc, 3, k_start, k_end, k_step, status)
!$omp parallel do
      !do k = 1, kmax
      !   do j = 1, jmax
      do k = k_start, k_end, k_step
         call xmp_template_ltog(t_desc, 3, k, g_k, status)
         do j = j_start, j_end, j_step
            call xmp_template_ltog(t_desc, 2, j, g_j, status)
      !--- 2020 Fujitsu end
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
               !--- 2020 Fujitsu
               !p(i,j,k)  = float(k-1)*float(k-1)/(float(kmax-1)     &
               !            *float(kmax-1))
               p(i,j,k)  = float(g_k-1)*float(g_k-1)/(float(kmax-1)     &
                           *float(kmax-1))
               !--- 2020 Fujitsu end
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

      !--- 2020 Fujitsu
      integer(4) :: status

! !$xmp reflect (p)
      call xmp_array_reflect(p_desc, status)
      !--- 2020 Fujitsu end
      DO loop = 1, nn

         gosa = 0.0
         !--- 2020 Fujitsu
! !$xmp loop (J,K) on t(*,J,K)
         call xmp_loop_schedule(2, jmax-1, 1, t_desc, 2, j_start, j_end, j_step, status)
         call xmp_loop_schedule(2, kmax-1, 1, t_desc, 3, k_start, k_end, k_step, status)
!$omp parallel do reduction(+:GOSA) private(S0,SS)
         !DO K = 2, kmax-1
         !   DO J = 2, jmax-1
         DO K = k_start, k_end, k_step
            DO J = j_start, j_end, j_step
         !--- 2020 Fujitsu
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
         
         !--- 2020 Fujitsu
! !$xmp loop (J,K) on t(*,J,K)
!$omp parallel do
         !DO K = 2, kmax-1
         !   DO J = 2, jmax-1
         DO K = k_start, k_end, k_step
            DO J = j_start, j_end, j_step
         !--- 2020 Fujitsu
               DO I = 2, imax-1
                  p(I,J,K) = wrk2(I,J,K)
               enddo
            enddo
         enddo
!$omp end parallel do

         !--- 2020 Fujitsu
! !$xmp reflect (p)
         call xmp_array_reflect(p_desc, status)
! !$xmp reduction (+:GOSA)
         call xmp_reduction_scalar(XMP_SUM, XMP_FLOAT, loc(GOSA), status)
         !--- 2020 Fujitsu end
      Enddo
! End of iteration
      return
      end
