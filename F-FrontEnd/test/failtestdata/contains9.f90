! if func is defined as integer in module, internal sub program as same
! name could not be defined.
! see also testdata/contains5.f90
      module mod1
        integer func
        contains
          function func ()
            func = 3
          end function
      end module

