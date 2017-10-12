      PROGRAM test_get_command
        INTEGER :: status, length
        CHARACTER(len=255) :: cmd
        CALL get_command()
        CALL get_command(cmd)
        CALL get_command(cmd, length)
        CALL get_command(cmd, length, status)
        WRITE (*,*) TRIM(cmd), length, status
      END PROGRAM test_get_command
