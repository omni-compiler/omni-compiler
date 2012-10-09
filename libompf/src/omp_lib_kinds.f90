! Copyright (c) 1997-2011 OpenMP Architecture Review Board.
! Permission to copy without fee all or part of this material is granted,
! provided the OpenMP Architecture Review Board copyright notice and
! the title of this document appear. Notice is given that copying is by
! permission of OpenMP Architecture Review Board.
!
module omp_lib_kinds
    integer, parameter :: omp_lock_kind = selected_int_kind( 10 )
    integer, parameter :: omp_nest_lock_kind = selected_int_kind( 10 )
    integer, parameter :: omp_sched_kind = selected_int_kind( 8 )
    integer(kind=omp_sched_kind), parameter :: omp_sched_static = 1
    integer(kind=omp_sched_kind), parameter :: omp_sched_dynamic = 2
    integer(kind=omp_sched_kind), parameter :: omp_sched_guided = 3
    integer(kind=omp_sched_kind), parameter :: omp_sched_auto = 4
end module omp_lib_kinds
