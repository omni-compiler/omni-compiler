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
        interface operator(.add.)
           module procedure add_point2D
        end interface
        interface vector
           function return(x)
             integer return,x
           end function return
           module procedure new_point2D
        end interface
        interface add_vector
           function add(x,y)
             integer x,y,add
           end function add
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

      program main
        use point2D_def
        implicit none

        type(point2D) :: a,b

        a = vector(1,2)
        b = vector(3,4)

        a = a .add. b

        print *, a%x, a%y
      end program main
