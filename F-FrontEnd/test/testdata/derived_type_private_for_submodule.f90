      MODULE m_dtpfs
        TYPE t
           INTEGER, PRIVATE :: a
           INTEGER :: b
        END TYPE t
        TYPE(t) :: v0 = t(a=1, b=2)
        TYPE(t) :: v1 = t(1, 2)

        INTERFACE
           MODULE FUNCTION f(a)
             INTEGER :: f
             INTEGER :: a
           END FUNCTION
        END INTERFACE
      END MODULE m_dtpfs

      SUBMODULE(m_dtpfs) sub
        TYPE(t) :: v2 = t(1,2)
      CONTAINS
        MODULE FUNCTION f(a)
          INTEGER :: f
          INTEGER :: a
          f = v0%a + a
        END FUNCTION
      END SUBMODULE sub

      PROGRAM MAIN
        USE m_dtpfs
        v0%b = 1
      END PROGRAM MAIN
