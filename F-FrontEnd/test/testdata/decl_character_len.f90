program test
    character(len=4) a(1), b*2, c(10)*3
    if(len(a) .ne. 4)then
        stop "a not 4"
    end if
    if(len(b) .ne. 2)then
        stop "b not 2"
    end if
    if(len(c) .ne.3)then
        stop "c not 3"
    end if
end

