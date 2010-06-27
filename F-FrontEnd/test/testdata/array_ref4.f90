      PROGRAM MAIN
        INTEGER,DIMENSION(1:3)  :: IX = (/1,2,3/)
        INTEGER,DIMENSION(1:3)  :: A  = (/2,4,6/)
        INTEGER,DIMENSION(1:5)  :: NX = (/0,0,0,0,0/)

        NX(IX(1:3)) = A(1:3)
        NX(1:3) = A(1:3)

        print *, NX
      END PROGRAM MAIN
