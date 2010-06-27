      function add(x,y)
        integer add,x,y
        add = x + y
      end function add

      function return(x)
        integer return,x
        return = x
      end function return

      module point2D_def
        type point2D
           integer :: x
           integer :: y
        end type point2D
        interface operator(+)
           module procedure add_point2D
        end interface
      contains
        function add_point2D(x,y) result(w)
          type(point2D) w
          type(point2D), intent(in) :: x, y

          w%x = x%x + y%x
          w%y = x%y + y%y
        end function add_point2D
        function sub_point2D(x,y) result(w)
          type(point2D) w
          type(point2D), intent(in) :: x, y

          w%x = x%x - y%x
          w%y = x%y - y%y
        end function sub_point2D
        function new_point2D(x,y) result(w)
          type(point2D) w
          integer x,y
          w%x = x
          w%y = y
        end function new_point2D
      end module point2D_def

      module point3D_def
        use point2D_def
        implicit none
        type point3D
           integer :: x
           integer :: y
           integer :: z
        end type point3D
        interface add_vector
           function add(x,y)
             integer x,y,add
           end function add
           module procedure add_point3D
           module procedure add_point2D
        end interface
        interface vector
           function return(x)
             integer return,x
           end function return
           module procedure new_point3D
           module procedure new_point2D
        end interface
      contains
        function add_point3D(x,y) result(w)
          type(point3D) w
          type(point3D), intent(in) :: x, y
          w%x = x%x + y%x
          w%y = x%y + y%y
          w%z = x%z + y%z
        end function add_point3D
        function new_point3D(x,y,z) result(w)
          type(point3D) w
          integer x,y,z
          w%x = x
          w%y = y
          w%z = z
        end function new_point3D
      end module point3D_def

      program main
        use point3D_def
        implicit none

        type(point2D) :: a,b
        type(point3D) :: p,q

        a = vector(1,2)
        b = vector(3,4)

        a = a + b
        a = add_vector(a, b)
        a = sub_point2D(a, b)

        print *, a%x, a%y

        p = vector(1,2,3)

        print *, p%x, p%y, p%z
      end program main
