      program main
      TYPE wall_type
          REAL,POINTER,DIMENSION(:) :: tmp_s
          REAL,POINTER,DIMENSION(:,:) :: ilw
      END TYPE
      TYPE(wall_type),POINTER,DIMENSION(:) :: wall
      REAL, ALLOCATABLE, DIMENSION(:,:) :: DLN
      INTEGER :: N, IW, IOR, IBND
      WALL(IW)%ILW(N,IBND) = SUM(- W_AXI * DLN(IOR,:) *              &
                           & WALL(IW)%ILW(:,IBND),1,DLN(IOR,:).LT.0.)
      end program main
