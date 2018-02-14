MODULE mod_issue492_2
  USE mod_issue492, only: nf_strerror

CONTAINS
  
  SUBROUTINE sub1()
    print*, trim(nf_strerror(0))
  END SUBROUTINE sub1
END MODULE mod_issue492_2

