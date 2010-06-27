program test

!
! implied do loop is described in doLoop*.f90
!

!#1
integer a1, a2, a3, a4
character a5 * 2
logical a6
real a7
double precision a8
complex a9
doublecomplex a10
parameter(i=1)
data a1 /1/
data a2, a3 /2, 3/
data a4 /i/
data a5 /'A5'/
data a6 /.TRUE./
data a7, a8 /1.0, 2.0/
data a9, a10 /(1.0, 2.0), (3.0, 4.0)/

!#2
integer b1(2:4), b2(4)
integer b3, b4
data b1 / 3*1 /
data b2 / 4*i /

!#3
character c1(2)
character c2
equivalence (c1(2), c2)
data c2    /'B'/

!#4
character d1*8
data d1(3:4) /'AA'/

end

