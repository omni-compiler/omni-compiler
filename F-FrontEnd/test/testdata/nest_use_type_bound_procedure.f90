      MODULE nest_use_tbp_mod_3
        USE use_tbp_mod_3
       CONTAINS
        SUBROUTINE sss(a)
          TYPE(tt), POINTER :: a
          PRINT *, a%p%v
        END SUBROUTINE sss
      END MODULE nest_use_tbp_mod_3

