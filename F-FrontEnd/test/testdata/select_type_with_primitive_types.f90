
program main
 contains
  subroutine sub(p)
    class(*) :: p 

    select type (p)
      type is(integer(kind=4))
        print*,'p is integer(kind=4)'
      type is(integer(kind=8))
        print*,'p is integer(kind=8)'
      type is(real)
        print*,'p is real'
      class default
        print*,'p is unknown type'
    end select
  end subroutine sub
end program main
