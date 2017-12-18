      INTEGER :: X = 1
      REAL :: Y = 2.5
      ASSOCIATE ( Y => X, Z => Y )
      PRINT *, Z
      END ASSOCIATE
      END
