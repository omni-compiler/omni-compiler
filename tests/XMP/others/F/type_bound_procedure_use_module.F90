#ifdef DEBUG
! NG by native compiler(gfortran 6.2.0/ifort 13.1.2).
      MODULE tbp_mod_type_bound_procedure_use_module
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE :: init => initT
           PROCEDURE :: print => printT
        END TYPE t

        TYPE, EXTENDS ( t ) :: tt
           INTEGER :: w
         CONTAINS
           PROCEDURE :: init => initTT ! override
           PROCEDURE :: print => printTT
        END TYPE tt

      CONTAINS
        SUBROUTINE printT(this)
          CLASS(t) :: this
          print *, v
        END SUBROUTINE printT
        SUBROUTINE printTT(this)
          CLASS(tt) :: this
          print *, v, w
        END SUBROUTINE printTT
        SUBROUTINE initT(this, v, w)
          ! initialize shape objects
          CLASS(t) :: this
          INTEGER :: v
          INTEGER, OPTIONAL :: w

          this%v = v
        END SUBROUTINE initT

        SUBROUTINE initTT(this, v, w)
          CLASS(tt) :: this
          INTEGER :: v
          INTEGER, OPTIONAL :: w
          this%v = v
          this%w = w
        END SUBROUTINE initTT
      END MODULE tbp_mod_type_bound_procedure_use_module

      PROGRAM main
        use tbp_mod_type_bound_procedure_use_module
        TYPE(t) :: a
        TYPE(tt) :: b
        call a%init(1,2)
        call a%print
        call b%init(3,4)
        call b%print
      END PROGRAM main
#else
print *, 'SKIPPED'
end
#endif

