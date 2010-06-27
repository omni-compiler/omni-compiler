program main
    integer,parameter::s4 = selected_int_kind(8)
    integer,parameter::s8 = selected_int_kind(10)
    integer,parameter::k4 = kind(1)
    integer,parameter::k8 = kind(1_8)
    integer,parameter::r8 = kind(1.0D0)
    integer(kind=s4) i_s4(2)
    integer(kind=s8) i_s8(2)
    integer(kind=k4) i_k4(2)
    integer(kind=k8) i_k8(2)
    integer(kind=r8) i_r8(2)

    if (kind(i_s4) /= 4) then
        print *, "invalid s4"
    end if
    if (kind(i_s8) /= 8) then
        print *, "invalid s8"
    end if
    if (kind(i_k4) /= 4) then
        print *, "invalid k4"
    end if
    if (kind(i_k8) /= 8) then
        print *, "invalid k8"
    end if
    if (kind(i_r8) /= 8) then
        print *, "invalid r8"
    end if
end program

