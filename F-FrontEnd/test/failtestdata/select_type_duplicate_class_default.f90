
program main
 contains
  subroutine sub(p)
    class(*) :: p 
    select type (p)
      class default
      class default
    end select
  end subroutine sub
end program main
