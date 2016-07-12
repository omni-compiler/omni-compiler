      PROGRAM main
        REAL, ALLOCATABLE :: A(:)[:]
        ALLOCATE ( A(10)[*])
        CRITICAL
          DEALLOCATE ( A )
        END CRITICAL
      END PROGRAM main
