! Copyright (c) 1997-2011 OpenMP Architecture Review Board.
! Permission to copy without fee all or part of this material is granted,
! provided the OpenMP Architecture Review Board copyright notice and
! the title of this document appear. Notice is given that copying is by
! permission of OpenMP Architecture Review Board.
!
       integer omp_lock_kind
       integer omp_nest_lock_kind
! this selects an integer that is large enough to hold a 64 bit integer
       parameter ( omp_lock_kind = selected_int_kind( 10 ) )
       parameter ( omp_nest_lock_kind = selected_int_kind( 10 ) )
       integer omp_sched_kind
! this selects an integer that is large enough to hold a 32 bit integer
       parameter ( omp_sched_kind = selected_int_kind( 8 ) )
       integer ( omp_sched_kind ) omp_sched_static
       parameter ( omp_sched_static = 1 )
       integer ( omp_sched_kind ) omp_sched_dynamic
       parameter ( omp_sched_dynamic = 2 )
       integer ( omp_sched_kind ) omp_sched_guided
       parameter ( omp_sched_guided = 3 )
       integer ( omp_sched_kind ) omp_sched_auto
       parameter ( omp_sched_auto = 4 )
