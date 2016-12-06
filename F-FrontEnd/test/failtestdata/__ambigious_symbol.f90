      ! NOTE:
      ! this code is not fail testdata,
      ! but is required by other fail testdata.
      module ambigious_symbol1
        integer :: n
      end module ambigious_symbol1

      module ambigious_symbol2
        integer :: n
      end module ambigious_symbol2

      module ambigious_symbol
        use ambigious_symbol1
        use ambigious_symbol2
      end module ambigious_symbol

      program main
        use ambigious_symbol
        n = 1
      end program main
