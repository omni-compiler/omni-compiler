      PROGRAM test_move_alloc
        INTEGER, PARAMETER :: N = 4
        INTEGER, ALLOCATABLE :: X(:), Y(:)
        ALLOCATE (X(N))
        X = 1
        CALL MOVE_ALLOC(X, Y)
        PRINT *, Y
      END PROGRAM test_move_alloc
