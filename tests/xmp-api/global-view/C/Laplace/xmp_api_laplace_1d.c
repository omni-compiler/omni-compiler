#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <xmp.h>
#include <xmp_api.h>

#define N1 100
#define N2 200
double u[N2][N1], uu[N2][N1];

// if np=8, p[2][4], j=N2->200/2=100, i=N1=100/4=25

// #pragma xmp nodes p[*]
// #pragma xmp template t[N2][N1]
// #pragma xmp distribute t[block][*] onto p
// #pragma xmp align u[j][i] with t[j][i]
// #pragma xmp align uu[j][i] with t[j][i]
// #pragma xmp shadow uu[1:1][0:0]

int main(int argc, char **argv)
{
  int i, j, k, niter = 10;
  double value = 0.0;
  int node_dims[1];
  xmp_desc_t p_desc, t_desc;
  xmp_desc_t u_desc, uu_desc;
  double *u_p, *uu_p;
  int lead_dims[2], uu_lda, u_lda;
  int j_init, j_cond, j_step, i_init, i_cond, i_step;
  long long int g_i, g_j;
  // int rank;

  xmp_api_init(argc,argv);
  // rank = xmpc_all_node_num();

  /* set up */
  /* #pragma xmp nodes p[*] */
  node_dims[0] = -1; /* nodes[*], DYNAMIC */
  p_desc = xmp_global_nodes(1,node_dims,FALSE);
  
  // printf("init node ...\n");
  // xmp_barrier();

  /* #pragma xmp template t[N2][N1] */
  t_desc = xmpc_new_template(p_desc, 2, (long long)N1, (long long)N2);
  /* #pragma xmp distribute t[block][block] onto p */
  xmp_dist_template_DUPLICATION(t_desc, 0);
  xmp_dist_template_BLOCK(t_desc, 1, 0);

  // printf("init template ...\n");
  // xmp_barrier();

  /* #pragma xmp align u[j][i] with t[j][i] */
  u_desc = xmpc_new_array(t_desc, XMP_DOUBLE, 2, (long long)N1, (long long)N2);
  xmp_align_array(u_desc, 0, 0, 0);
  xmp_align_array(u_desc, 1, 1, 0);
  xmp_allocate_array(u_desc,(void **)&u_p);

  // printf("init array u ...\n");
  // xmp_barrier();

  /* #pragma xmp align uu[j][i] with t[j][i] */
  uu_desc = xmpc_new_array(t_desc, XMP_DOUBLE, 2, (long long)N1, (long long)N2);
  xmp_align_array(uu_desc, 0, 0, 0);
  xmp_align_array(uu_desc, 1, 1, 0);

  /* #pragma xmp shadow uu[1:1][0:0] */
  xmp_set_shadow(uu_desc,0,0,0);
  xmp_set_shadow(uu_desc,1,1,1);

  xmp_allocate_array(uu_desc,(void **)&uu_p);

  // printf("init array uu ...\n");
  // xmp_barrier();

  xmp_array_lead_dim(u_desc,lead_dims);
  u_lda = lead_dims[0];
  xmp_array_lead_dim(uu_desc,lead_dims);
  uu_lda = lead_dims[0];

#define U(j,i) (u_p[u_lda*(j)+(i)])
#define UU(j,i) (uu_p[uu_lda*((j)+1)+((i)+1)])


  // #pragma xmp loop (j,i) on t[j][i]
  xmpc_loop_schedule(0,N2,1,t_desc,1,&j_init,&j_cond,&j_step);
  //  xmpc_loop_schedule(0,N1,1,t_desc,0,&i_init,&i_cond,&i_step);
  for(j = j_init/*0*/; j < j_cond/*N2*/; j+=j_step/*j++*/){
    for(i = 0; i < N1; i++){
      U(j,i) = 0.0;  // u[j][i] = 0.0;
      UU(j,i) = 0.0; // uu[j][i] = 0.0;
    }
  }
  
  xmp_barrier();

  // #pragma xmp loop (j,i) on t[j][i]
  xmpc_loop_schedule(1,N2-1,1,t_desc,1,&j_init,&j_cond,&j_step);
  //  xmpc_loop_schedule(1,N1-1,1,t_desc,0,&i_init,&i_cond,&i_step);
  for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
    for(i = 1; i < N1-1; i ++){
      g_i = i; //xmp_template_ltog(t_desc,0,i,&g_i);
      xmp_template_ltog(t_desc,1,j,&g_j);
      // u[j][i] = sin((double)i/N1*M_PI) + cos((double)j/N2*M_PI);
      U(j,i) = sin((double)g_i/N1*M_PI) + cos((double)g_j/N2*M_PI);
  }

  for(k = 0; k < niter; k++){

    if(xmpc_all_node_num() == 0) printf("iter =%d\n",k);

    // #pragma xmp loop (j,i) on t[j][i]
    for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
      for(i = 1; i < N1-1; i ++){
	UU(j,i) = U(j,i); // uu[j][i] = u[j][i];
      }

    // #pragma xmp reflect (uu)
    xmp_array_reflect(uu_desc);

    //#pragma xmp loop (j,i) on t[j][i]
    for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
      for(i = 1; i < N1-1; i ++){
	U(j,i) = (UU(j-1,i)+UU(j+1,i)+UU(j,i-1)+UU(j,i+1))/4.0;
	// u[j][i] = (uu[j-1][i] + uu[j+1][i] + uu[j][i-1] + uu[j][i+1])/4.0;
      }

    value = 0.0;
    // #pragma xmp loop (j,i) on t[j][i] reduction(+:value)
    for(j = j_init/*1*/; j < j_cond /*N2-1*/; j += j_step /*j++*/)
      for(i = 1; i < N1-1; i ++){
	// value += fabs(uu[j][i] - u[j][i]);
	value += fabs(UU(j,i) - U(j,i));
      }
    xmp_reduction_scalar(XMP_SUM, XMP_DOUBLE, (void *)&value);

    //#pragma xmp task on p[0][0]
    if(xmpc_all_node_num() == 0)
      printf("Verification = %20.16f\n", value);
  }

  return 0;
}
