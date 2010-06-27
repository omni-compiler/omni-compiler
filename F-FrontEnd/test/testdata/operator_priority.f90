program main

    double precision a, b, c, d
    double precision ra1, ra2, rb1, rb2
    a = 1D0
    b = 2D0
    c = 3D0
    d = 4D0

    !ra1's AST must equal to ra2's AST 
    ra1 = -a * b * c * d
    ra2 = -(((a * b) * c) * d)

    !rb1's AST must equal to rb2's AST 
    rb1 = -a / b / c / d
    rb2 = -(((a / b) / c) / d)

    !rc1's AST must equal to rc2's AST 
    rc1 = -a ** b
    rc2 = -(a ** b)

    print *,ra1, ra2
    print *,rb1, rb2
    print *,rc1, rc2

end
