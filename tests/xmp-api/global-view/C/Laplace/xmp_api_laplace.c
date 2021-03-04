#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <xmp.h>
#include <xmp_api.h>

#define N1 100
#define N2 200
double u[N2][N1], uu[N2][N1];

// #pragma xmp nodes p[*][4]
// #pragma xmp template t[N2][N1]
// #pragma xmp distribute t[block][block] onto p
// #pragma xmp align u[j][i] with t[j][i]
// #pragma xmp align uu[j][i] with t[j][i]
// #pragma xmp shadow uu[1:1][1:1]

int main(int argc, char **argv)
{
  int i, j, k, niter = 10;
  double value = 0.0;
  int node_dims[2];
  xmp_desc_t p_desc, t_desc;
  xmp_desc_t u_desc, uu_desc;
  double *u_p, *uu_p;

  xmp_api_init(argc,argv);

  /* set up */
  /* #pragma xmp nodes p[*][4] */
  node_dims[1] = 0; node_dims[0] = 4; /* nods[*][4], DYNAMIC */
  p_desc = xmp_global_nodes(2,node_dims,FALSE);
  
  /* #pragma xmp template t[N2][N1] */
  t_desc = xmpc_new_template(p_desc, 2, (long)N1, (long)N2);
  /* #pragma xmp distribute t[block][block] onto p */
  xmp_dist_template_BLOCK(t_desc, 0, 0);
  xmp_dist_template_BLOCK(t_desc, 1, 1);

  /* #pragma xmp align u[j][i] with t[j][i] */
  u_desc = xmpc_new_array(desc_t, XMP_DOUBLE, 2, (long)N1, (long)N2);
  xmp_align_array(u_desc, 0, 1, 0);
  xmp_align_array(u_desc, 1, 0, 0);
  xmp_allocate_array(u_desc,&u_p);

  /* #pragma xmp align uu[j][i] with t[j][i] */
  uu_desc = xmpc_new_array(desc_t, XMP_DOUBLE, 2, (long)N1, (long)N2);
  xmp_align_array(uu_desc, 0, 1, 0);
  xmp_align_array(uu_desc, 1, 0, 0);

  /* #pragma xmp shadow uu[1:1][1:1] */
  xmp_set_shadow(uu_desc,0,1,1);
  xmp_set_shadow(uu_desc,1,1,1);
  xmp_allocate_array(uu_desc,&uu_p);
  
  xmp_array_lda(u_desc,&u_lda);
  xmp_array_lda(uu_desc,&uu_lda);
#define U(j,i) (*(u_pp + u_lda*(j)+(i)))
#define UU(j,i) (*(uu_pp + uu_lda*(j)+(i)))

  // #pragma xmp loop (j,i) on t[j][i]
  xmpc_loop_schedule(0,N2,1,t_desc,1,&j_init,&j_cond,&j_step);
  xmpc_loop_schedule(0,N1,1,t_desc,0,&i_init,&i_cond,&i_step);
  for(j = j_init/*0*/; j < j_cond/*N2*/; j+=j_step/*j++*/){
    for(i = i_init/*0*/; i < i_cond /*N1*/; i+=i_step/*i++*/){
      U(j,i) = 0.0;  // u[j][i] = 0.0;
      UU(j,i) = 0.0; // uu[j][i] = 0.0;
    }
  }

  // #pragma xmp loop (j,i) on t[j][i]
  xmpc_loop_schedule(1,N2-1,1,t_desc,1,0,&j_init,&j_cond,&j_step);
  xmpc_loop_schedule(1,N1-1,1,t_desc,0,0,&i_init,&i_cond,&i_step);
  for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
    for(i = i_init/*1*/; i < i_cond /* N1-1*/; i += i_step /*i++*/){
      xmp_array_ltog(u_desc,0,&g_i);
      xmp_array_ltog(u_desc,1,&g_j);
      // u[j][i] = sin((double)i/N1*M_PI) + cos((double)j/N2*M_PI);
      U(j,i) = sin((double)g_i/N1*M_PI) + cos((double)g_j/N2*M_PI);
  }

  for(k = 0; k < niter; k++){

    if(xmpc_all_node_num() == 0) printf("iter =%d\n",k);

    // #pragma xmp loop (j,i) on t[j][i]
    for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
      for(i = i_init/*1*/; i < i_cond /* N1-1*/; i += i_step /*i++*/){
	UU(j,i) = U(j,i); // uu[j][i] = u[j][i];
      }

    // #pragma xmp reflect (uu)
    xmp_array_reflect(uu_desc);

    //#pragma xmp loop (j,i) on t[j][i]
    for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
      for(i = i_init/*1*/; i < i_cond /* N1-1*/; i += i_step /*i++*/){
	U(j,i) = (UU(j-1,i)+UU(j+1,i)+UU(j,i-1)+UU(j,i+1))/4.0;
	// u[j][i] = (uu[j-1][i] + uu[j+1][i] + uu[j][i-1] + uu[j][i+1])/4.0;
      }
  
    // #pragma xmp loop (j,i) on t[j][i] reduction(+:value)
    for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
      for(i = i_init/*1*/; i < i_cond /* N1-1*/; i += i_step /*i++*/){
	// value += fabs(uu[j][i] - u[j][i]);
	value = fabs(UU(j,i) - U(j,i));
      }
    xmp_reduction();

  //#pragma xmp task on p[0][0]
  if(xmpc_all_node_num() == 0)
    printf("Verification = %20.16f\n", value);

  return 0;
}
