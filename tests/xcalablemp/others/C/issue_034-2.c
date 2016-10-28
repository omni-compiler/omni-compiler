#include <stdio.h>

struct ST1 {
  int x, y;
};
struct ST2 {
  struct ST1 X, Y;
} a = {{1,3},{2,4}};

int main(void)
{
    if (a.X.x==1 && a.X.y==3 && a.Y.x==2 && a.Y.y==4)
      printf("PASS\n");
    else
      printf("FAIL : %d, %d, %d, %d not 1, 3, 2, 4\n", a.X.x, a.X.y, a.Y.x, a.Y.y);
    return 0;
}

