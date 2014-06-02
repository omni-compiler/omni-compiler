subroutine sub
    integer,parameter::n=3
    integer,pointer::p(:)
    integer,target,save:: a(0:n)
    !$omp threadprivate(a)
    p=>a
    a = 1
end subroutine
