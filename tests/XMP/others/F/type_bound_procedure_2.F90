#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
          MODULE m_type_bound_procedure_2
            TYPE :: t
              INTEGER :: v
            CONTAINS
              PROCEDURE,PASS,PUBLIC :: show => show
              PROCEDURE,PASS,PUBLIC :: inc => inc
              PROCEDURE,PASS,PUBLIC :: check => check
            END TYPE t
          CONTAINS
            SUBROUTINE show(a)
              CLASS(t) :: a
              PRINT *, a%v
            END SUBROUTINE show
            FUNCTION inc(a)
              CLASS(t) :: a
              a%v = a%v + 1
              inc = a%v
            END FUNCTION
            subroutine check(a)
              CLASS(t) :: a
              if (a%v.eq.2) then
                print *, 'PASS'
              else
                print *, 'NG : ', a%v, ', should be 2'
                call exit(1)
              endif
            END subroutine
          END MODULE m_type_bound_procedure_2

          PROGRAM MAIN
            USE m_type_bound_procedure_2
            INTEGER :: i
            CLASS(t), POINTER :: a
            TYPE(t), TARGET :: b
            b = t(v=1)
            a => b
!           CALL a%show()
            i = a%inc()
!           CALL a%show()
            CALL a%check()
!           PRINT *, i
          END PROGRAM MAIN
#else
print *, 'SKIPPED'
end
#endif

