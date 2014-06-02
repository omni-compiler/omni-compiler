program main
    integer a, b, i
    a = 1
    b = 10
    !$omp atomic
    a = a + 5
    !$omp atomic
    a = min(a, b)

    print *, a, b
end program

