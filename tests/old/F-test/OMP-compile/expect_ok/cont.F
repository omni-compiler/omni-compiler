      subroutine sub
      integer x, i1, i2
C$OMP PARALLEL
C$OMP+IF(6+(-1)*X.LE.0)
C$OMP+DEFAULT(SHARED)
C$OMP+PRIVATE(I1,I2)
      nx = 1
c$omp end parallel
      end subroutine

