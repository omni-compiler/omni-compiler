#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

      PROGRAM main
        USE use_procedure_decls

        INTERFACE
          FUNCTION u(arg)
            INTEGER :: u
            INTEGER :: arg
          END FUNCTION u
          SUBROUTINE v()
          END SUBROUTINE v
        END INTERFACE

        a => u
        b => v

      END PROGRAM main
#else
END
#endif
