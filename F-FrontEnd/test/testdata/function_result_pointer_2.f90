      module mod
        real,target :: tmp
       contains
        subroutine sub()
          real a
          a = 3.0
          print *, f(a)
          f(a) = 6.0   ! ポインタ関数結果に対して代入をする。
          print *, f(a)
        end subroutine sub
        function f(key) result(res)
          real,intent(in) :: key
          real,pointer :: res
          tmp = key + 2.0
          res => tmp
        end function f
      end module mod
      !
      program main
        use mod
        real a
        a = 3.0
        print *, f(a)
        f(a) = 6.0   ! ポインタ関数結果に対して代入をする。
        print *, f(a)
      end program main

