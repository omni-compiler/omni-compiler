      PROGRAM test_get_command_argument
        INTEGER :: status, length
        CHARACTER(len=255) :: val
        CALL get_command_argument(0)
        CALL get_command_argument(0, val)
        CALL get_command_argument(0, val, length)
        CALL get_command_argument(0, val, length, status)
        WRITE (*,*) TRIM(val), length, status
      END PROGRAM test_get_command_argument
