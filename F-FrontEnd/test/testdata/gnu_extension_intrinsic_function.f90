      MODULE m_gnu_extension
        INTERFACE
          MODULE FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
        END INTERFACE
      CONTAINS
        SUBROUTINE sub()
          COMPLEX :: r
          r = COMPLEX(1.0,2.0)
        END SUBROUTINE SUB
      END MODULE m_gnu_extension

      SUBMODULE(m_gnu_extension) subm
      CONTAINS
        SUBROUTINE sub2()
          COMPLEX :: r
          r = COMPLEX(1.0,2.0)
        END SUBROUTINE sub2
      END SUBMODULE subm

      PROGRAM main
        COMPLEX :: r
        r = COMPLEX(1.0,2.0)
        BLOCK
          COMPLEX :: r
          r = COMPLEX(1.0,2.0)
        END BLOCK
      END PROGRAM main
