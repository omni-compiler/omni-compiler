      MODULE m
        PRIVATE :: e, f, g, h
        ENUM, BIND(C)
          ENUMERATOR a
          ENUMERATOR b
          ENUMERATOR :: c = 10
          ENUMERATOR e, f
          ENUMERATOR :: g = 100, h = 1000, i
        END ENUM
      END MODULE m

      PROGRAM main; USE m; END PROGRAM main
