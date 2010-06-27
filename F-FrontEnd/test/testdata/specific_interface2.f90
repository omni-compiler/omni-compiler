      module mod1
        private
        public :: id

        interface id
           module procedure my_id
        end interface
      contains
        function my_id(a)
          integer my_id, a
          my_id = a
        end function my_id
      end module mod1

      module mod2
        private
        public :: id

        interface id
           module procedure my_id
        end interface
      contains
        function my_id(a)
          character my_id, a
          my_id = a
        end function my_id
      end module mod2

      module mod3
        private
        public :: id

        interface id
           module procedure my_id
        end interface
      contains
        function my_id(a)
          real my_id, a
          my_id = a
        end function my_id
      end module mod3

      program main
        use mod1
        use mod2
        use mod3

        print *, id("a")
        print *, id(3)
        print *, id(3.0)
      end program main
