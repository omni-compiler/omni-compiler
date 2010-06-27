      subroutine init_locks

      integer lmor
      parameter(lmor=92700)

cc    integer (kind=omp_lock_kind) tlock(lmor)
c$    integer*8 tlock(lmor)

      integer i

c$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i)
c$    do i=1,lmor
c$      call omp_init_lock(tlock(i))
c$    end do

      return
      end
