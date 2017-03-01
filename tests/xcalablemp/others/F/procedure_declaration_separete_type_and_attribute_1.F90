#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        PROCEDURE(REAL) :: g
        POINTER :: g
        REAL ret
        g => f
        ret = g()
      CONTAINS
        FUNCTION f()
          REAL f
          PRINT *, 'PASS'
          f = 1.0
        END FUNCTION
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
