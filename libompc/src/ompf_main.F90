! $TSUKUBA_Release: Omni OpenMP Compiler 3 $
! $TSUKUBA_Copyright:
!  PLEASE DESCRIBE LICENSE AGREEMENT HERE
!  $
PROGRAM main
    external ompf_init
    external ompf_main
    external ompf_terminate

    CALL ompf_init()
    CALL ompf_main()
    CALL ompf_terminate(0);
END PROGRAM

