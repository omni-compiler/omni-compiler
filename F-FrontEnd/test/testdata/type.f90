      program main
        type point3d
          real x
          real y
          real z
        end type
        type point3d2
          type(point3d) p1
          type(point3d) :: p2
        end type
      end
