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
        print*,'class is point'
      type is(color_point)
        print*,'class is color_point'
      class default
        print*,'class default'
    end select
  end subroutine sub1


  subroutine sub2(p)
    class(point) :: p

    select type (a=>p)
       class is(point)
        PRINT*,'class is point',a%x
       class is(color_point)
         print*,'class is color_point',a%color
       class default
         print*,'default'
    end select

  end subroutine sub2
end module
