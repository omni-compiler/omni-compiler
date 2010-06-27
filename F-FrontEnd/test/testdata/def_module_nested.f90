      module m0
        implicit none
        character(LEN=1) :: a
      end module m0

      module m1
        use m0
        implicit none
        private
        public :: a
        integer :: b
      end module m1

      program main
        use m1
        implicit none
        real b

        a = "c"
      end program main
