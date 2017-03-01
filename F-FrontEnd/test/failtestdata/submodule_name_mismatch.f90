      MODULE m_name_mismatch
        INTERFACE
          MODULE SUBROUTINE sub()
          END SUBROUTINE
        END INTERFACE
      END MODULE m_name_mismatch

      SUBMODULE(m_name_mismatch) subm
      END SUBMODULE subhoge

      PROGRAM main
        USE m_name_mismatch
      END PROGRAM main
