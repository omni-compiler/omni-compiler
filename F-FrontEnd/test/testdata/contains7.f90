! public, private and modules complicated case
      module inf_mod
        private
        public :: func
        contains
          function func()
            integer func
            func = 5
          end function func

      end module inf_mod
      module sup_mod
        use inf_mod

        integer,public :: func

      end module sup_mod
