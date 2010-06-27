      function strfunc(x)
        character(10) :: strfunc
        integer :: x
        strfunc = "ABCDEFG"
      end function
      program main
        character(10) :: string
        character(5)  :: substring
        character(10) :: strfunc
        substring = string(2:6)
        string = strfunc()
      end
