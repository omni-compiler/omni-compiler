#if defined(__GNUC__) && (4 < __GNUC__ || 4 == __GNUC__ && 7 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
               MODULE m_type_bound_procedure
                 TYPE :: t
                   INTEGER :: v
                 CONTAINS
                   PROCEDURE,PASS :: show => show
                   PROCEDURE,PASS :: check => check
                 END TYPE t
               CONTAINS
                 SUBROUTINE show(a)
                   CLASS(t) :: a
                   PRINT *, a%v
                 END SUBROUTINE show
                 SUBROUTINE check(a)
                   CLASS(t) :: a
                   PRINT *, 'NG : t%check called.'
                   call exit(1)
                 END SUBROUTINE check
               END MODULE m_type_bound_procedure

               MODULE mm_type_bound_procedure
                 USE m_type_bound_procedure
                 TYPE, EXTENDS(t) :: tt
                   INTEGER :: u
                 CONTAINS
                   PROCEDURE,PASS :: show => show2
                   PROCEDURE,PASS :: check => check2
                 END TYPE tt
               CONTAINS
                 SUBROUTINE show2(a)
                   CLASS(tt) :: a
                   PRINT *, a%u
                 END SUBROUTINE show2
                 SUBROUTINE check2(a)
                   CLASS(tt) :: a
                   if (a%u.eq.2) then
                     PRINT *, 'PASS'
                   else
                     PRINT *, 'NG : ', a%u, ', shoule be 2'
                     call exit(1)
                   endif
                 END SUBROUTINE check2
               END MODULE mm_type_bound_procedure

               PROGRAM MAIN
                 USE mm_type_bound_procedure
                 CLASS(t), POINTER :: a
                 TYPE(tt),TARGET :: b = tt(v=1, u=2)
                 a => b
!                CALL a%show()
                 CALL a%check()
               END PROGRAM MAIN
#else
print *, 'SKIPPED'
end
#endif
