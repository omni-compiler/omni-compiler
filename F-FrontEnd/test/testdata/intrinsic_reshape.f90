      program test_reshape
        integer, dimension(2,3) :: a, b
        integer, dimension(2) :: shape2d
        integer, dimension(6) :: dat

        dat = 3

        shape2d = (/2,3/)

        a = reshape( (/ 1, 2, 3, 4, 5, 6 /), (/ 2, 3 /))
        b = reshape( (/ 0, 7, 3, 4, 5, 8 /), (/ 2, 3 /))

        a = reshape( (/ 1, 2, 3, 4, 5, 6 /), shape2d)
        b = reshape( (/ 0, 7, 3, 4, 5, 8 /), shape2d)

        !dat = shape(a) ! What test do you do at this line?

        a = reshape( dat, shape(a))
        b = reshape( dat, shape(b))

      end program
