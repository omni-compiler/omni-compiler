    interface
      function f()
        class(*), pointer :: f
      endfunction f
    end interface
    select type (p => f())
      type is(integer)
        print*,'type is integer'
      class default
        print*,'default'
    end select
    end

    function f()
      class(*), pointer :: f
      allocate(integer :: f)
    end function
