     1 PROGRAM MAIN
     2   INTEGER,ALLOCATABLE:: a(:)
     3 
     4   BLOCK
     5     ALLOCATE(a(1000))
     6     DEALLOCATE(a)
     7   END BLOCK
     8 
     9 END PROGRAM MAIN
