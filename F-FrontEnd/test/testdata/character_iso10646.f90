      SUBROUTINE create_date_string(string)
        INTRINSIC date_and_time,selected_char_kind
        INTEGER,PARAMETER :: ucs4 = selected_char_kind("ISO_10646")
        CHARACTER(1,UCS4),PARAMETER :: nen=CHAR(INT(Z'5e74'),UCS4), & !year
             gatsu=CHAR(INT(Z'6708'),UCS4), & !month
             nichi=CHAR(INT(Z'65e5'),UCS4) !day
        CHARACTER(len= *, kind= ucs4) string
        INTEGER values(8)
        CALL date_and_time(values=values)
        WRITE(string,1) values(1),nen,values(2),gatsu,values(3),nichi
1       FORMAT(I0,A,I0,A,I0,A)
      END SUBROUTINE create_date_string

      PROGRAM main
        INTEGER,PARAMETER :: ucs4 = selected_char_kind("ISO_10646")
        CHARACTER(len=100, kind= ucs4) :: string

        CALL create_date_string(string)
        OPEN(6, encoding="utf-8")
        PRINT *, string
      END PROGRAM main
        
