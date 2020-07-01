#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
      SUBMODULE(m_submodule_01) m_submodule_02
      CONTAINS
        MODULE PROCEDURE func1
          TYPE(t), POINTER :: v
          TYPE(t), TARGET :: u
          COMPLEX :: r
          r = COMPLEX(para1, para1*10)
          v => u
          v%i = REAL(r)
          v%k = IMAG(r)
          CALL v%p1(func1)
        END PROCEDURE
      END SUBMODULE m_submodule_02
#else
SUBROUTINE SUB1
  PRINT *, 'SKIPPED'
END SUBROUTINE
#endif

