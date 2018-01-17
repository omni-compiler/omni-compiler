      MODULE tbp_mod
        TYPE t
           INTEGER :: v
         CONTAINS
           PROCEDURE :: init => initT
        END TYPE t

        TYPE, EXTENDS ( t ) :: tt
           INTEGER :: w
         CONTAINS
           PROCEDURE :: init => initTT ! override
        END TYPE tt

      CONTAINS
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
      END MODULE tbp_mod
