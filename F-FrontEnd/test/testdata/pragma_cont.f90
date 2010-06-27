    integer a
    !$omp parallel&
    !$omp private(a)
    print *,a
    !$omp end parallel
    end

