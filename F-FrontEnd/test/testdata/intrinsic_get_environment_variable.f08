      PROGRAM test_get_environment_variable
        INTEGER :: status, length
        CHARACTER(len=255) :: val
        CALL get_environment_variable("HOME")
        CALL get_environment_variable("HOME", val)
        CALL get_environment_variable("HOME", val, length)
        CALL get_environment_variable("HOME", val, length, status)
        CALL get_environment_variable("HOME", val, length, status, .TRUE.)
        WRITE (*,*) TRIM(val), length, status
      END PROGRAM test_get_environment_variable
