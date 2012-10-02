      module extfile_generic_procedure2
        private
        public :: g
        interface g
          module procedure f
        end interface g
      contains
        function f(a)
          real :: f, a
        end function f
      end module extfile_generic_procedure2
