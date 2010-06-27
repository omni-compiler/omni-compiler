      program main
        if(logicalfunc())then
            print *,"true"
        end if

        contains
            logical function logicalfunc()
                logicalfunc = .true.
            end function
      end
