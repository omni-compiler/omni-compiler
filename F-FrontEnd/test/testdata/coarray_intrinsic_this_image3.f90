      PROGRAM main
        REAL A (10, 20) [10, 0:9, 0:*]
        INTEGER image
        image = THIS_IMAGE(A,1)
        WRITE (*,*) image
      END PROGRAM main
