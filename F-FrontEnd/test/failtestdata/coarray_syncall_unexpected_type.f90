      PROGRAM main
        INTEGER :: v = 1, u = 2
        CHARACTER :: c = 'a', d = 'b'
        SYNC ALL (STAT=c, ERRMSG=c)
        SYNC ALL (STAT=v, ERRMSG=v)
      END PROGRAM main
