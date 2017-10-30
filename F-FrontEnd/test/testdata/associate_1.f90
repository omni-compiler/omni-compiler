      REAL :: A = 1.0
      REAL :: X = 1.0, Y = 2.3, THETA = 3.14
      ASSOCIATE ( Z => EXP(-(X**2+Y**2)) * COS(THETA) )
        PRINT *, A+Z
        PRINT *, A-Z
      END ASSOCIATE
      END
