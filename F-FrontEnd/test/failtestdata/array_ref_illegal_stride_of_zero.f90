      program main
        real, dimension(10) :: vector
        real, dimension(5)  :: subvector
        subvector(1:5:0) = vector(1:10:0)
        ! illegal stride of Zero
      end
