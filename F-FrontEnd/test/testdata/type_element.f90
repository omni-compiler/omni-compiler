      program main
        type nline
          integer no
          integer, dimension(80) :: buffer
        end type
        type(nline) line
        line%no = 10
      end
