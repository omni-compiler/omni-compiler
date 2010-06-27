      program main
        integer, dimension(:,:), allocatable :: m
        integer, pointer :: p
        integer, target :: i
        integer :: res
        allocate (m(3,3))
        deallocate (m, stat=res)
        nullify (p)
        p => i
      end
