      MODULE m_dtpfe
        TYPE t
           INTEGER, PRIVATE :: a
           INTEGER :: b
        END TYPE t
        TYPE(t) :: v0 = t(a=1, b=2)
        TYPE(t) :: v1 = t(1, 2)
      END MODULE m_dtpfe

      PROGRAM MAIN
        USE m_dtpfe

        TYPE,EXTENDS(t) :: tt
           INTEGER :: a
        END type TT

        v0%b = 1
      END PROGRAM MAIN
