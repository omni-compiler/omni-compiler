      MODULE m
        INTEGER, POINTER :: v
      END MODULE m
      PROGRAM MAIN
        BLOCK
          USE m
          VOLATILE :: v
          ALLOCATE(v)
          v = 2
          PRINT *, v
        END BLOCK
        BLOCK
          USE m
          PRINT *, v
        END BLOCK
      END PROGRAM

