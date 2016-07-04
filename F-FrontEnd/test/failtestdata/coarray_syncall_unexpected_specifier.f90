      PROGRAM main
        INTEGER :: v = 1, u = 2
        CHARACTER :: c = 'a', d = 'b'
        SYNC ALL (STAT=v, UNEXPECTED=v, ERRMSG=c)
      END PROGRAM main
