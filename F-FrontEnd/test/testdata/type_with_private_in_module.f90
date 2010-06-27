      module mod1
        type point3d
          private
          real x
          real y
          real z
        end type
        type, public ::  point3d2
          private
          type(point3d) p1
        end type
      end module
