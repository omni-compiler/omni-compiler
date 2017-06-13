#include <stdio.h>

#pragma xmp nodes p[2][2]

#pragma xmp template t[10][10]
#pragma xmp distribute t[block][block] onto p

double b[10][10];
#pragma xmp align b[i][j] with t[i][j]
#pragma xmp shadow b[0][*]


int main(){
#pragma xmp reflect (b)

#pragma xmp task on p[0][0]
  {
    printf("PASS\n");
  }

}
