      PROGRAM read3
0011  FORMAT(I3)
        INTEGER*8 l
        INTEGER :: UNIT = 1

        READ (UNIT = *, FMT = *) l
        READ (UNIT, FMT = *) l
        READ (UNIT = 0011, FMT = *) l
        READ (UNIT = *, FMT = 0011) l
        READ (UNIT = 0011, FMT = 0011) l
    END PROGRAM read3
