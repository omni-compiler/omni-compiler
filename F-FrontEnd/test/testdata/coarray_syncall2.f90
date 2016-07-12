      PROGRAM main
        INTEGER :: v = 1
        CHARACTER :: c = 'a'
        SYNC ALL (STAT=v, ERRMSG=c)
      END PROGRAM main
