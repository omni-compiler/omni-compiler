#pragma xmp nodes p(2)
#pragma xmp template t(0:9)
#pragma xmp distribute t(block) onto p
int a1[10], a2[10];
#pragma xmp align [i] with t[i] :: a1, a2

