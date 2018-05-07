      MODULE use_procedure_decls
        USE procedure_decls

#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)

       CONTAINS
        SUBROUTINE check()
          integer :: i
          a => p
          i = p (1)
          b => q
          if(i.eq.10) then
            PRINT *, 'PASS 1'
          else
            PRINT *, 'NG 1'
            CALL EXIT(1)
          end if
          CALL b
        END SUBROUTINE check
        FUNCTION p(arg)
          INTEGER :: p
          INTEGER :: arg
          p = arg * 10
        END FUNCTION p
        SUBROUTINE q()
          PRINT *, 'PASS 2'
        END SUBROUTINE q
#endif
      END MODULE use_procedure_decls
