#include <xmp.h>
#include <stdio.h>
#pragma xmp nodes p[2][2]
#pragma xmp template t[10][10]
#pragma xmp distribute t[block][cyclic] onto p
int a[10][10];
#pragma xmp align a[i][j] with t[i][j]

int main()
{
  int me = xmpc_node_num(), flag = 0, global_i;
  int dim = 1;

  xmp_array_lbound_global(xmp_desc_of(a), dim, &global_i);
  
  if(me == 0 || me == 1){
    if(global_i != 0){
      flag = 1;
    }
  }
  else if(me == 2 || me == 3){
    if(global_i != 5){
      flag = 1;
    }
  }

  dim = 2;
  xmp_array_lbound_global(xmp_desc_of(a), dim, &global_i);

  if(me == 0 || me == 2){
    if(global_i != 0){
      flag = 1;
    }
  }
  else if(me == 1 || me == 3){
    if(global_i != 1){
      flag = 1;
    }
  }
#pragma xmp reduction(+:flag)

  if(flag == 0){
    if(me == 0)
      printf("PASS\n");
    return 0;
  }

  if(me == 0)
    printf("Error\n");
  
  return 1;
}

