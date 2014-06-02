MODULE MM
  INTEGER, TARGET, ALLOCATABLE :: pa_a(:)

  CONTAINS
    SUBROUTINE GET_THDPRV_A(pp_a, tp_a)
      IMPLICIT NONE
      integer, target:: pp_a
      EXTERNAL ompf_thread_lock
      INTEGER :: ompf_get_num_threads
      EXTERNAL ompf_get_num_threads
      EXTERNAL ompf_thread_unlock
      LOGICAL :: ompf_is_master
      EXTERNAL ompf_is_master
      INTEGER, POINTER :: tp_a
      INTEGER :: ompf_get_thread_num
      EXTERNAL ompf_get_thread_num

      CALL ompf_thread_lock()
      IF ((.NOT.allocated(pa_a))) THEN
       ALLOCATE (pa_a(ompf_get_num_threads()))
      END IF
      CALL ompf_thread_unlock()
      IF (ompf_is_master()) THEN
       tp_a => pp_a
      ELSE
       tp_a => pa_a(ompf_get_thread_num())
      END IF
    END SUBROUTINE
END MODULE

SUBROUTINE sub2(pp_a)
 USE MM
 IMPLICIT NONE
 integer, target:: pp_a
 INTEGER, POINTER :: tp_a
 INTEGER :: a
 COMMON /cmn/ a
 call GET_THDPRV_A(pp_a, tp_a)

 tp_a = 111
END SUBROUTINE sub2

SUBROUTINE ompf_func_0(pp_a)
 EXTERNAL sub2
 integer a
 COMMON /cmn/ a

 CALL sub2(pp_a)
END SUBROUTINE ompf_func_0

SUBROUTINE ompf_dop_1(ompf_func)
 EXTERNAL ompf_func
 INTERFACE
  SUBROUTINE ompf_do_parallel(p__narg, ompf_func)
   INTEGER :: p__narg
   INTERFACE
    SUBROUTINE ompf_func()
    END SUBROUTINE ompf_func
   END INTERFACE

  END SUBROUTINE ompf_do_parallel
 END INTERFACE

 CALL ompf_do_parallel(0, ompf_func)
END SUBROUTINE ompf_dop_1

SUBROUTINE sub1()
 USE MM
 INTEGER, POINTER :: tp_a
 EXTERNAL ompf_func_0
 EXTERNAL sub2
 INTEGER, TARGET :: a
 COMMON /cmn/ a
 INTERFACE
  SUBROUTINE ompf_dop_1(ompf_func, pp_a)
   integer pp_a
   INTERFACE
    SUBROUTINE ompf_func()
    END SUBROUTINE ompf_func
   END INTERFACE

  END SUBROUTINE ompf_dop_1
 END INTERFACE

 call GET_THDPRV_A(a, tp_a)
 tp_a = 222
 CALL ompf_dop_1(ompf_func_0, a)
END SUBROUTINE sub1

