      program main
        use two_path_struct, only: t1 => t
        use two_path_struct, only: t2 => t
        type(t1) :: a
        type(t2) :: b
        b = a
      end program main
