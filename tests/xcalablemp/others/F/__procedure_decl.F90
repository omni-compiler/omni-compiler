      MODULE procedure_decls
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

        PRIVATE :: f, sub
        PROCEDURE(f), POINTER :: a => f
        PROCEDURE(sub), POINTER :: b => sub
        PROCEDURE(), POINTER :: c
       CONTAINS
        FUNCTION f(arg)
          INTEGER :: f
          INTEGER :: arg
        END FUNCTION f
        SUBROUTINE sub()
          PRINT *, 'NG 2'
          CALL EXIT(1)
        END SUBROUTINE sub
#endif
      END MODULE procedure_decls

      MODULE m__procedure_decl
        USE procedure_decls
      END MODULE m__procedure_decl
