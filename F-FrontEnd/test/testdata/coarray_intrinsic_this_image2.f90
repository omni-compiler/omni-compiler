      PROGRAM main
        REAL A (10, 20) [10, 0:9, 0:*]
        INTEGER,DIMENSION(0:2) :: image
        image(0:2) = THIS_IMAGE(A)
        WRITE (*,*) image
      END PROGRAM main
