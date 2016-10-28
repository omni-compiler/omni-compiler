      PROGRAM MAIN
        TYPE :: type1
          INTEGER :: v
        END TYPE type1
        BLOCK
          TYPE :: type2
            INTEGER :: v
          END TYPE type2
          BLOCK
            TYPE :: type3
              INTEGER :: v
            END TYPE type3
            BLOCK
              TYPE :: type4
                INTEGER :: v
              END TYPE type4
              TYPE(type1) :: a
              TYPE(type2) :: b
              TYPE(type3) :: c
              TYPE(type4) :: d
            END BLOCK
          END BLOCK
        END BLOCK
      END PROGRAM MAIN
