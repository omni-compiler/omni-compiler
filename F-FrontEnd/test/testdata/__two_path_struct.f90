      module two_path_struct
        type t
          integer :: n
        end type t
      end module two_path_struct

      module two_path_struct1; use two_path_struct; end module two_path_struct1
      module two_path_struct2; use two_path_struct; end module two_path_struct2
