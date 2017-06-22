      MODULE m_dtpp
        TYPE t
           PRIVATE
           INTEGER :: a = 3
           INTEGER, PUBLIC :: b
        END TYPE t
        TYPE(t) :: v0 = t(a=1, b=2)
        TYPE(t) :: v1 = t(1, 2)
      END MODULE m_dtpp

      PROGRAM MAIN
        USE m_dtpp
        TYPE(t) :: v2 = t(b=2)
        v0%b = 1
      END PROGRAM MAIN
