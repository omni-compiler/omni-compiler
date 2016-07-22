      PROGRAM main
        USE ISO_FORTRAN_ENV
        TYPE tt
           INTEGER(ATOMIC_INT_KIND) i
           LOGICAL(ATOMIC_LOGICAL_KIND) l
        END type tt

        INTEGER(ATOMIC_INT_KIND) :: I[*]
        LOGICAL(ATOMIC_LOGICAL_KIND) :: L[*]
        INTEGER(ATOMIC_INT_KIND) A (10, 20) [10, 0:9, 0:*]
        LOGICAL(ATOMIC_LOGICAL_KIND) B (10, 20) [10, 0:9, 0:*]

        INTEGER :: u
        LOGICAL :: v

        TYPE(tt) :: t[*]

        CALL ATOMIC_DEFINE(I, 1)
        CALL ATOMIC_DEFINE(I[2], 1)

        CALL ATOMIC_REF(u, I)
        CALL ATOMIC_REF(u, I[2])

        CALL ATOMIC_DEFINE(L, .TRUE.)
        CALL ATOMIC_DEFINE(L[2], .FALSE.)

        CALL ATOMIC_REF(v, L)
        CALL ATOMIC_REF(v, L[2])

        CALL ATOMIC_DEFINE(A(1,1)[1,0,0], 1)
        CALL ATOMIC_DEFINE(B(1,1)[1,0,0], .FALSE.)

        CALL ATOMIC_REF(u, A(1,1)[1,0,0])
        CALL ATOMIC_REF(v, B(1,1)[1,0,0])

        CALL ATOMIC_DEFINE(t%i, 1)
        CALL ATOMIC_DEFINE(t%l, .TRUE.)

        CALL ATOMIC_REF(u, t%i)
        CALL ATOMIC_REF(v, t%l)

        CALL ATOMIC_DEFINE(t[2]%i, 1)
        CALL ATOMIC_DEFINE(t[3]%l, .TRUE.)

        CALL ATOMIC_REF(u, t[2]%i)
        CALL ATOMIC_REF(v, t[3]%l)

      END PROGRAM main
