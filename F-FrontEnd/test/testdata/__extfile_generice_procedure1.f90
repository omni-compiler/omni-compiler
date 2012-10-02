      module extfile_generic_procedure1
        private
        public :: g
        interface g
          module procedure f
        end interface g
      contains
        function f(a)
          integer :: f, a
        end function f
      end module extfile_generic_procedure1
