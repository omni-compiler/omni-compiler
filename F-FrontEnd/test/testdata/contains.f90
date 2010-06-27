      module mod
        contains
          function func ()
           func = hoge()
           contains
             function hoge()
               hoge = 1
             end function
          end function
      end module

      program main
        contains
          function func_main ()
            func_main = 1
          end function
      end program

