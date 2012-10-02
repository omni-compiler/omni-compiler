      module m1
        integer :: i
      end module m1

      program main
        use m1
      contains
        function i()
          real :: i
        end function i
      end program main
