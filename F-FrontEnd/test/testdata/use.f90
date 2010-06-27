      module mod1
      end module

      module mod2
      end module

      module mod3
        integer i
      end module

      subroutine sub1
        use mod1
      end subroutine

      function func2 ()
        use mod2
        integer func2
        func2 = 0
      end function

      program main
        use mod3
        call sub1
        i = func2
      end program
