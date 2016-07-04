      PROGRAM main
        REAL, ALLOCATABLE :: A(:)[:]
        CRITICAL
          ALLOCATE ( A(10)[*])
        END CRITICAL
      END PROGRAM main
