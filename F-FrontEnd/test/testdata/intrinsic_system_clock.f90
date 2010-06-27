          PROGRAM test_system_clock
            INTEGER :: count, count_rate, count_max
            CALL SYSTEM_CLOCK(count, count_rate, count_max)
            WRITE(*,*) count, count_rate, count_max
          END PROGRAM
