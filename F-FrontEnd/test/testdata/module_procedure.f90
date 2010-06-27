      module POINT3D
        type point
           real x, y, z
        end type point

        interface eq
          module procedure POINT3D_EQ
        end interface 

        contains

        function POINT3D_EQ (p1, p2)
          logical :: POINT3D_EQ
          type(point), intent(in) :: p1
          type(point), intent(in) :: p2
          POINT3D_EQ = &
            p1%x == p2%x .AND. &
            p1%y == p2%y .AND. &
            p1%z == p2%z
        end function
      end module
