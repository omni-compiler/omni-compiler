      program main
        LOGICAL  :: is_exist
        LOGICAL  :: is_opened
        INTEGER  :: number
        LOGICAL  :: is_named
        CHARACTER (10) :: name
        CHARACTER (10) :: access
        CHARACTER (10) :: sequential
        CHARACTER (10) :: direct
        CHARACTER (10) :: form
        CHARACTER (10) :: formatted
        CHARACTER (10) :: unformatted
        INTEGER :: recl
        INTEGER :: next_rec
        CHARACTER (10) :: blank
        CHARACTER (10) :: position
        CHARACTER (10) :: action
        CHARACTER (10) :: read
        CHARACTER (10) :: write
        CHARACTER (10) :: read_write
        CHARACTER (10) :: delim
        CHARACTER (10) :: pad

        OPEN(UNIT=5, &
             IOSTAT=ios, &
             ERR=99, &
             FILE='filename', &
             STATUS='REPLACE', &
             ACCESS='SEQUENTIAL', &
             FORM='FORMATTED', &
             BLANK='ZERO', &
             POSITION='APPEND', &
             ACTION='READWRITE', &
             DELIM='NONE', &
             PAD='YES')

        INQUIRE(UNIT=5, &
             IOSTAT=ios, &
             ERR=99, &
             EXIST=is_exist, &
             OPENED=is_opened, &
             NUMBER=number, &
             NAMED=is_named, &
             ACCESS=access, &
             SEQUENTIAL=sequential, &
             DIRECT=direct, &
             FORM=form, &
             FORMATTED=formatted, &
             UNFORMATTED=unformatted, &
             RECL=recl, &
             NEXTREC=next_rec, &
             BLANK=blank, &
             POSITION=position, &
             ACTION=action, &
             READ=read, &
             WRITE=write, &
             READWRITE=read_write, &
             DELIM=delim, &
             PAD=pad)

        CLOSE(UNIT=5, &
             IOSTAT=ios, &
             ERR=99, &
             STATUS='KEEP')
        RETURN
99      WRITE(*,*) "err"
      end program main
