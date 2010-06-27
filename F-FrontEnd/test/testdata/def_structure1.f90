type tt
    integer a
    real b
end type tt

type t2
    type(tt) t
    integer k
end type t2

type(tt)::t
type(t2)::x

t%a = 1
t%b = 2.
x%t%a = 10
end
