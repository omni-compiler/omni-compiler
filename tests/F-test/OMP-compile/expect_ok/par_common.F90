      program EMBAR

      implicit none

      double precision x, q, qq
      integer          nk, nq
      logical          verified, timers_enabled
      external         randlc, timer_read
      double precision randlc, timer_read
      character*15     size

      parameter (nk = 1, nq = 10)
      common/sharedq/ q(0:nq-1)

!$omp parallel
      call timer_clear(1)
      if (timers_enabled) call timer_clear(2)
      if (timers_enabled) call timer_clear(3)
!$omp end parallel
      call timer_start(1)

      end

