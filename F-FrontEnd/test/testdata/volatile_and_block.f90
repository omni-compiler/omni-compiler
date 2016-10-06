  1       PROGRAM MAIN
  2         INTEGER :: a
  3         a = 1
  4         PRINT *, a
  5         BLOCK
  6           INTEGER :: a
  7           VOLATILE :: a
  8           a = 2
  9           PRINT *, a
 10         END BLOCK
 11         PRINT *, a
 12       END PROGRAM MAIN

