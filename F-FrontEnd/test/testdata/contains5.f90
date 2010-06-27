! if public, private attribute is specified, then type integer
! declaration is ok
      module mod1
        integer, public :: func
        contains
          function func ()
            func = 3
          end function
      end module

