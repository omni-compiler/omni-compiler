program main
    integer, external::omp_get_num_threads
    print *, omp_get_num_threads()
    !$omp master
    print *, "in master"
    !$omp end master
end program

