module select_type
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

    cname: select type (p)
      class is(point) cname
        print*,'class is point'
      type is(color_point) cname
        print*,'class is color_point'
      class default cname
        print*,'class default'
    end select cname

    select type (assoc_name=>p)
      class is(point)
        print*,'class is point',assoc_name%x
      class is(color_point)
        print*,'class is color_point',assoc_name%color
      class default
        print*,'default'
    end select

  end subroutine sub1
end module select_type
