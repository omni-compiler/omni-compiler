      module m
        use extfile_generic_procedure1
        interface g
          module procedure f
        end interface g
      contains
        function f(a)
          complex :: f,a
        end function f
      end module m
