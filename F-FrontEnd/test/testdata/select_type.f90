module shape_mod
  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type

contains

  subroutine sub1(p)
    class(point) :: p 

    select type (p)
      CLASS IS (shape)
        PRINT*,'SHAPE'
    CLASS IS (square)
        PRINT*,'SQUARE'
    CLASS DEFAULT
        PRINT*,'DEFAULT'
    end select
  end subroutine sub1
end module
