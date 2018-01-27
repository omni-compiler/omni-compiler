      TYPE :: t
        INTEGER :: v
      END TYPE t
      TYPE :: tt
        TYPE(t), POINTER :: u
      END TYPE tt

      TYPE(tt) :: a = tt(NULL())
      END
     
