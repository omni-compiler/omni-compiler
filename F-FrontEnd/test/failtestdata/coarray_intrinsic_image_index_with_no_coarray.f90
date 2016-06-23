      PROGRAM main
        REAL A (10, 20)
        INTEGER index
        index = IMAGE_INDEX(A,(/1,0,0/))
        WRITE (*,*) index
      END PROGRAM main
