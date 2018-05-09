#include <stdio.h>
#include <unistd.h>
#include <xmp.h>

#pragma xmp nodes p[2]
int* b;
#pragma xmp template t[:]

#pragma xmp distribute t[block] onto p
#pragma xmp align b[i] with t[i]

void test(int* b){
#pragma xmp align b[i] with t[i]
	
#pragma xmp loop(i) on t[i]
  for(int i=0;i<10;i++)
    printf("[%d] b[%d] = %d\n", xmpc_node_num(), i, b[i]);

}

int main()
{
#pragma xmp template_fix[block] t[10]
  b = (int*)xmp_malloc(xmp_desc_of(b), 10);
	
#pragma xmp loop (i) on t[i]
  for(int i=0;i<10;i++)
    b[i] = i;
	
#pragma xmp loop(i) on t[i]
  for(int i=0;i<10;i++)
    printf("[%d] b[%d] = %d\n", xmpc_node_num(), i, b[i]);
		
  sleep(1);
  printf("\n");
  sleep(1);

  test(b);
	
  return 0;
} 
