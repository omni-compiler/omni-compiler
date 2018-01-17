      MODULE tbp_mod_3
        TYPE t
          INTEGER :: v
         CONTAINS
          PROCEDURE, PASS :: s => s
        END TYPE t
       CONTAINS
        SUBROUTINE s(self)
          CLASS(t) :: self
          self%v = self%v + 1
        END SUBROUTINE s
      END MODULE tbp_mod_3
