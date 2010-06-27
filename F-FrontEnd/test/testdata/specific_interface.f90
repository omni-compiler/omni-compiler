      module inf_mod
        private
        public :: xc, inf_xc

        interface xc
           module procedure inf_xc
        end interface
        contains
          function inf_xc(a)
            integer inf_xc,a
            inf_xc = 5
          end function inf_xc
      end module inf_mod

      module sup_mod
        use inf_mod

        private
        public :: xc, sup_xc, inf_xc

        interface xc
           module procedure sup_xc
        end interface
      contains
        function sup_xc(a)
          real sup_xc,a
          sup_xc = 10
        end function sup_xc
      end module sup_mod

      program main
        use sup_mod

        print *, xc(1)
        print *, inf_xc(1)
        print *, xc(3.0)
        print *, sup_xc(3.0)
      end program main
