#define TRUE 1
#define FALSE 0
#include <stdio.h>
#include <string.h>
#include <xmp_api.h>
#include "xmp.h"

int a_test[20];
xmp_desc_t a_desc;
int (*a_p)[20];
int img_dims[1];

float b_test[4][10];
xmp_desc_t b_desc;
float (*b_p)[4][10];

double c_test[2][10][10];
xmp_desc_t c_desc;
double (*c_p)[2][10][10];

int return_val = 0;

void initialize(int me){
  int i, j, m, n, t = me * 100;
  
  for(i=0;i<20;i++){
    (*a_p)[i] = i + t;
    a_test[i] = (*a_p)[i];
  }
  
  for(i=0;i<4;i++){
    for(j=0;j<10;j++){
      (*b_p)[i][j] = 10*i + j + t;
      b_test[i][j] = (*b_p)[i][j];
    }
  }
	
  for(i=0;i<2;i++){
    for(j=0;j<10;j++){
      for(m=0;m<10;m++){
	      (*c_p)[i][j][m] = 100*i + 10*j + m + t;
	      c_test[i][j][m] = (*c_p)[i][j][m];
      }
    }
  }

}

void communicate_1(int me){
  xmp_array_section_t *a_section;
  xmp_sync_all(NULL);


  if(me == 1){
    int tmp[100];
    long tmp_dims[1];
    xmp_local_array_t *tmp_local;
    xmp_array_section_t *tmp_l_section;

    a_section = xmp_new_array_section(1);
    tmp_l_section = xmp_new_array_section(1);
    xmp_array_section_set_triplet(a_section,0,2,5,1);
    xmp_array_section_set_triplet(tmp_l_section,0,3,5,2);

    tmp_dims[0] = 100;    
    tmp_local = xmp_new_local_array(sizeof(int), 1, tmp_dims, (void **)&tmp);


    img_dims[0] = 0;
    // tmp[3:5:2] = a[2:5]:[0]; // get
    xmp_coarray_get_local(img_dims,a_desc,a_section,tmp_local,tmp_l_section);
    (*a_p)[3] = tmp[3]; (*a_p)[5] = tmp[5]; (*a_p)[7] = tmp[7];
    (*a_p)[9] = tmp[9]; (*a_p)[11] = tmp[11];
		xmp_free_array_section(a_section);
		xmp_free_array_section(tmp_l_section);
  }

  if(me == 0){
    int tmp[50];
    long tmp_dims[1];
    xmp_local_array_t *tmp_local;
    xmp_array_section_t *tmp_l_section;

    a_section = xmp_new_array_section(1);
    tmp_l_section = xmp_new_array_section(1);
    xmp_array_section_set_triplet(a_section,0,0,2,1);
    xmp_array_section_set_triplet(tmp_l_section,0,8,2,2);

    tmp_dims[0] = 50;    

    tmp_local = xmp_new_local_array(sizeof(int), 1, tmp_dims, (void **)&tmp);

    img_dims[0] = 1;

    tmp[8] = 999; tmp[10] = 1000;

    //a[0:2]:[1] = tmp[8:2:2]; // put
    xmp_coarray_put_local(img_dims,a_desc,a_section,tmp_local,tmp_l_section);
		xmp_free_array_section(a_section);
		xmp_free_array_section(tmp_l_section);
  }

  if(me == 1){
    a_test[3] = 2; a_test[5] = 3; a_test[7] = 4;
    a_test[9] = 5; a_test[11] = 6; 
    a_test[0] = 999; a_test[1] = 1000;
  }
  
  xmp_sync_all(NULL);
}

void check_1(int me){
  int i, flag = TRUE;
  
  for(i=0; i<20; i++){
    if( (*a_p)[i] != a_test[i] ){
      flag = FALSE;
      printf("[%d] (*a_p)[%d] check_1 : fall\ta[%d] = %d (True value is %d)\n",
	     me, i, i, (*a_p)[i], a_test[i]);
    }
  }
  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_1 : PASS\n", me);
  else return_val = 1;
}

void communicate_2(int me){
  xmp_sync_all(NULL);

  if(me == 0){
    xmp_array_section_t *b_remote_section;
    xmp_array_section_t *b_local_section;

    b_remote_section = xmp_new_array_section(2);
		xmp_array_section_set_triplet(b_remote_section,0,2,1,1);
		xmp_array_section_set_triplet(b_remote_section,1,2,3,2);

    b_local_section = xmp_new_array_section(2);
		xmp_array_section_set_triplet(b_local_section,0,1,1,1);
		xmp_array_section_set_triplet(b_local_section,1,1,3,1);

    img_dims[0] = 1;
    //b[2][2:3:2]:[1] = b[1][1:3];  // put
    xmp_coarray_put(img_dims,b_desc,b_remote_section,b_desc,b_local_section);

		xmp_free_array_section(b_remote_section);
		xmp_free_array_section(b_local_section);

    b_remote_section = xmp_new_array_section(2);
		xmp_array_section_set_triplet(b_remote_section,0,2,1,1);
		xmp_array_section_set_triplet(b_remote_section,1,1,3,2);

    b_local_section = xmp_new_array_section(2);
		xmp_array_section_set_triplet(b_local_section,0,3,1,1);
		xmp_array_section_set_triplet(b_local_section,1,2,3,1);

    //b[3][2:3] = b[2][1:3:2]:[1];  // get
    xmp_coarray_get(img_dims,b_desc,b_remote_section,b_desc,b_local_section);

		xmp_free_array_section(b_remote_section);
		xmp_free_array_section(b_local_section);
  }
  if(me == 0){
    b_test[3][2] = 121; b_test[3][3] = 123; b_test[3][4] = 125;
  }
  if(me == 1){
    b_test[2][2] = 11;
    b_test[2][4] = 12;
    b_test[2][6] = 13;
  }
  
  xmp_sync_all(NULL);
}

void check_2(int me){
  xmp_sync_all(NULL);
  int i, j, flag = TRUE;
  
  for(i=0;i<4;i++){
    for(j=0;j<10;j++){
      if( (*b_p)[i][j] != b_test[i][j] ){
	flag = FALSE;
	printf("[%d] (*b_p)[%d][%d] check_2 : fall\tb[%d][%d] = %.f (True value is %.f)\n",
	       me, i, j, i, j, (*b_p)[i][j], b_test[i][j]);
      }
    }
  }
  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_2 : PASS\n", me);
  else return_val = 1;
}

void communicate_3(int me){
  xmp_sync_all(NULL);

  if(me == 1){
    xmp_array_section_t *c_remote_section;
    xmp_array_section_t *c_local_section;

    c_remote_section = xmp_new_array_section(3);
		xmp_array_section_set_triplet(c_remote_section,0,1,1,1);
		xmp_array_section_set_triplet(c_remote_section,1,2,2,3);
		xmp_array_section_set_triplet(c_remote_section,2,0,1,2);

    c_local_section = xmp_new_array_section(3);
		xmp_array_section_set_triplet(c_local_section,0,1,1,1);
		xmp_array_section_set_triplet(c_local_section,1,1,2,5);
		xmp_array_section_set_triplet(c_local_section,2,1,1,1);

    img_dims[0] = 0;
    // c[1][2:2:3][0:1:2]:[0] = c[1][1:2:5][1];   // put
    xmp_coarray_put(img_dims,c_desc,c_remote_section,c_desc,c_local_section);
		xmp_free_array_section(c_remote_section);
		xmp_free_array_section(c_local_section);
  }

  if(me == 0){
    double tmp[5][5];
    long tmp_dims[2];
    xmp_array_section_t *c_remote_section;
    xmp_local_array_t *tmp_local;
    xmp_array_section_t *tmp_l_section;

    tmp_l_section = xmp_new_array_section(2);
    xmp_array_section_set_triplet(tmp_l_section,0,0,3,2);
    xmp_array_section_set_triplet(tmp_l_section,1,0,1,1);

    tmp_dims[0] = 5;
    tmp_dims[1] = 5;
    tmp_local = xmp_new_local_array(sizeof(double), 2, tmp_dims, (void **)&tmp);

    c_remote_section = xmp_new_array_section(3);
		xmp_array_section_set_triplet(c_remote_section,0,0,1,1);
		xmp_array_section_set_triplet(c_remote_section,1,1,3,2);
		xmp_array_section_set_triplet(c_remote_section,2,2,1,1);

    img_dims[0] = 1;

    // tmp[0:3:2][0] = c[0][1:3:2][2]:[1];       // get
    xmp_coarray_get_local(img_dims,c_desc,c_remote_section,tmp_local,tmp_l_section);

		xmp_free_array_section(c_remote_section);
		xmp_free_array_section(tmp_l_section);
    
    (*c_p)[0][1][0] = tmp[0][0];
    (*c_p)[0][3][0] = tmp[2][0];
    (*c_p)[0][5][0] = tmp[4][0];
  }
  
  if(me == 0){
    c_test[1][2][0] = 111 + 100;
    c_test[1][5][0] = 161 + 100;
    c_test[0][1][0] = 112; c_test[0][3][0] = 132; c_test[0][5][0] = 152;
  }
  
  xmp_sync_all(NULL);
}

void check_3(int me){
  xmp_sync_all(NULL);
  int i, j, m, flag = TRUE;

  for(i=0;i<2;i++){
    for(j=0;j<10;j++){
      for(m=0;m<10;m++){
	if( (*c_p)[i][j][m] != c_test[i][j][m] ){
	  flag = FALSE;
	  printf("[%d] (*c_p)[%d][%d][%d] check_3 : fall\tc[%d][%d][%d] = %.f (True value is %.f)\n",
		 me, i, j, m, i, j, m, (*c_p)[i][j][m], c_test[i][j][m]);
	}
      }
    }
  }
  xmp_sync_all(NULL);
  if(flag == TRUE)   printf("[%d] check_3 : PASS\n", me);
  else return_val = 1;
}

int main(int argc, char *argv[]){
  long a_dims[1];
  long b_dims[2];
  long c_dims[3];
  xmp_api_init(argc,argv);
  int me = xmpc_this_image();

  a_dims[0] = 20;
  a_desc = xmp_new_coarray(sizeof(int), 1, a_dims, 1, img_dims, (void **)&a_p);

  b_dims[0] = 4;
  b_dims[1] = 10;
  b_desc = xmp_new_coarray(sizeof(float), 2,b_dims,1,img_dims,(void **)&b_p);

  c_dims[0] = 2;
  c_dims[1] = 10;
  c_dims[2] = 10;
  c_desc = xmp_new_coarray(sizeof(double), 3,c_dims,1,img_dims,(void **)&c_p);

  initialize(me);

  communicate_1(me);
  check_1(me);

  communicate_2(me);
  check_2(me);

  communicate_3(me);
  check_3(me);

#pragma xmp barrier
#pragma xmp reduction(MAX:return_val)  
  xmp_api_finalize();
  return return_val;
}
