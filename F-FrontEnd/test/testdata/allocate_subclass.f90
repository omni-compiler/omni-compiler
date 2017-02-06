      PROGRAM main
        TYPE t
          INTEGER :: v
        END TYPE t

        TYPE, EXTENDS(t) :: tt
          INTEGER :: u
        END TYPE tt

        CLASS(t), ALLOCATABLE :: a

        ALLOCATE(t :: a)

        ALLOCATE(tt :: a)

      END PROGRAM main
