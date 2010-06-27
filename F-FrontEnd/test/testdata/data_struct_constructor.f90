type tt
    integer a
    double precision b
    character c*10
end type

type(tt)::a
data a /tt(1, 2.0D1, "ccc")/
print *,a
end

