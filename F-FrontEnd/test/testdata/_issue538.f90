MODULE issue538_mod

IMPLICIT NONE

CONTAINS

  SUBROUTINE sub1(fill_value)
    REAL, INTENT(IN), OPTIONAL :: fill_value
    REAL, ALLOCATABLE :: recv_buffer(:,:)
    INTEGER :: collector_size(10)
    INTEGER :: nlev, global_size

    ALLOCATE(recv_buffer(nlev, MERGE(global_size, SUM(collector_size(:)), PRESENT(fill_value))))

  END SUBROUTINE sub1
END MODULE issue538_mod
