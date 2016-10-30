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
      class is(point)
        PRINT*,'class is point'
      type is(color_point)
        PRINT*,'class is color_point'
      class default
        PRINT*,'class default'
    end select
  end subroutine sub1
end module
