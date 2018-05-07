#pragma xmp nodes p(NUM_NODES)
#pragma xmp template t(0:N-1)
#pragma xmp distribute t(block) onto p
#pragma xmp align array[i] with t(i)
