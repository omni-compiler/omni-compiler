#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        USE use_procedure_decls

        CALL check
      END PROGRAM main
#else
PRINT *, 'SKIPPED'
END
#endif
