PROGRAM p1
  CALL sub1
END PROGRAM p1

SUBROUTINE sub1(data_length)
  INTEGER, INTENT(IN) :: data_length
  CHARACTER(LEN=data_length+2) :: data
  CHARACTER(LEN=2) :: data2

END SUBROUTINE