program main
    integer jump
    ASSIGN 10 TO jump
    GO TO jump (10, 20)
    stop "should not be here"
 10 continue
 20 print *, "ok"
end program main
