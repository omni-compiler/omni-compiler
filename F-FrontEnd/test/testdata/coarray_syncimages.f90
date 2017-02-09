      PROGRAM main
        INTEGER :: v = 1
        CHARACTER :: c = 'a'
        SYNC IMAGES (*)
        SYNC IMAGES (1)
        SYNC IMAGES (1, STAT=v, ERRMSG=c)
        SYNC IMAGES ((/1,2,3/))
      END PROGRAM main
