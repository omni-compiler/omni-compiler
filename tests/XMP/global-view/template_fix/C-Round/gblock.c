#include <xmp.h>
#include <stdio.h>
#include <stdlib.h>     
static const int N=1000;
#pragma xmp nodes p(*)
#pragma xmp template t(:)
#pragma xmp distribute t(gblock(*)) onto p
int i,s=0,procs,remain;
int *a,*m,result=0;
#pragma xmp align a[i] with t(i)

int main(void)
{
  procs = xmp_num_nodes();
  m = (int *)malloc(sizeof(int) * procs);
  remain = N;

  for(i=0;i<procs-1;i++){
    m[i] = remain/2;
    remain -= m[i];
  }
  m[procs-1] = remain;

#pragma xmp template_fix(gblock(m)) t(0:N-1)
  a = (int *)xmp_malloc(xmp_desc_of(a), N);

#pragma xmp loop on t(i)
   for(i=0;i<N;i++)
     a[i] = i;

#pragma xmp loop on t(i) reduction(+:s)
   for(i=0;i<N;i++)
     s += a[i];
   
   if(s != 499500)
     result = -1;

#pragma xmp reduction(+:result)
#pragma xmp task on p(1)
   {
     if(result == 0){
       printf("PASS\n");
     }
     else{
       fprintf(stderr, "ERROR\n");
       exit(1);
     }
   }
   
   free (a);
   free (m);
   return 0;
}
