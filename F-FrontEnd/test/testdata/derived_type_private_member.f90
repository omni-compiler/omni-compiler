      MODULE m_dtpm
        TYPE t
           INTEGER, PRIVATE :: a
           INTEGER :: b
        END TYPE t
        TYPE(t) :: v0 = t(a=1, b=2)
        TYPE(t) :: v1 = t(1, 2)
      END MODULE m_dtpm

      PROGRAM MAIN
        USE m_dtpm
        v0%b = 1
      END PROGRAM MAIN
