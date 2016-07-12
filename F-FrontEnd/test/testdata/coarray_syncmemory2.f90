      PROGRAM main
        INTEGER :: v = 1
        CHARACTER :: c = 'a'
        SYNC MEMORY (STAT=v, ERRMSG=c)
      END PROGRAM main
