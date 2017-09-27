       PROGRAM main
         INTEGER, DIMENSION(:,:), POINTER :: a
         INTEGER, DIMENSION(1:25), TARGET :: b
         a(1:5,1:5) => b
       END PROGRAM main
