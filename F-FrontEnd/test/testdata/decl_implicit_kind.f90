program main
    implicit integer(8) (a)
    implicit integer(kind=8) (b, c)
    implicit real(8) (d)
    implicit real(kind=8) (e, f)
    implicit complex(8) (g)
    implicit complex(kind=8) (h, i)
    dimension c(10)
    dimension f(10)
    dimension i(10)
    a = X'FFFFFFFFFFFFFFF'
    b = X'FFFFFFFFFFFFFFF'
    c(1) = X'FFFFFFFFFFFFFFF'
    d = dabs(d)
    e = dabs(e)
    f(1) = dabs(f(1))
    g = dabs(aimag(g))
    h = dabs(aimag(h))
    i(1) = dabs(aimag(i(1)))
end
