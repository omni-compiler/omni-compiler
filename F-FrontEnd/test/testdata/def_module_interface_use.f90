      module inferior_module
        integer,parameter::p1 = 1
        interface
           function array_func()
             integer, dimension(5) :: array_func
           end function array_func
        end interface
        interface array_func_spec
           function array_func2()
             integer, dimension(5) :: array_func2
           end function array_func2
        end interface
      end module inferior_module

      function superior_func() result(w)
        use inferior_module
        integer a, w

        a = maxval(array_func())

        a = p1

        w = 1
      end function superior_func
