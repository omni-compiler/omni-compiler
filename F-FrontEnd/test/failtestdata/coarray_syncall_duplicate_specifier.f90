      PROGRAM main
        INTEGER :: v = 1, u = 2
        CHARACTER :: c = 'a', d = 'b'
        SYNC ALL (STAT=v, STAT=u, ERRMSG=c)
        SYNC ALL (STAT=v, ERRMSG=c, ERRMSG=d)
      END PROGRAM main
