#include <stdio.h>

struct st{
    int x,y;
};
int main(void)
{
    struct st a = (struct st){2,3};
    if (a.x==2 && a.y==3)
      printf("PASS\n");
    else
      printf("FAIL : %d, %d not 2, 3\n", a.x, a.y);
    return 0;
}

