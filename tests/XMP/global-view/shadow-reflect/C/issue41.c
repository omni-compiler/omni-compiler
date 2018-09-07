#include <stdio.h>

typedef struct test {
  double v;
} test_t;

#pragma xmp template t(0:9,0:9)
#pragma xmp nodes p(2,2)
#pragma xmp distribute t(block, block) onto p
test_t a[10][10];
#pragma xmp align a[i][j] with t(j,i)
#pragma xmp shadow a[1][1]

int main()
{
#pragma xmp reflect (a)  async(1)
#pragma xmp wait_async(1)

#pragma xmp task on p(1,1)
  {
    printf("PASS\n");
  }
  
  return 0;
}
