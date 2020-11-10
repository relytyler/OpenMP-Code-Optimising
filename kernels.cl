#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

// typedef struct
// {
//   float speeds[NSPEEDS];
// } t_speed;

void reduce(                                          
   local  float*,
   local  int*,
   global float*,
   global int*,
   int timestep);

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[(3 * ny * nx) + ii + jj* nx] - w1) > 0.f
      && (cells[(6 * ny * nx) + ii + jj* nx] - w2) > 0.f
      && (cells[(7 * ny * nx) + ii + jj* nx] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[(1 * ny * nx) + ii + jj* nx] += w1;
    cells[(5 * ny * nx) + ii + jj* nx] += w2;
    cells[(8 * ny * nx) + ii + jj* nx] += w2;
    /* decrease 'west-side' densities */
    cells[(3 * ny * nx) + ii + jj* nx] -= w1;
    cells[(6 * ny * nx) + ii + jj* nx] -= w2;
    cells[(7 * ny * nx) + ii + jj* nx] -= w2;
  }
}


kernel void propagate(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      int nx, int ny)
{
  // /* get column and row indices */
  // int ii = get_global_id(0);
  // int jj = get_global_id(1);

  // /* determine indices of axis-direction neighbours
  // ** respecting periodic boundary conditions (wrap around) */
  // int y_n = (jj + 1) % ny;
  // int x_e = (ii + 1) % nx;
  // int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  // int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  // /* propagate densities from neighbouring cells, following
  // ** appropriate directions of travel and writing into
  // ** scratch space grid */
  // tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
  // tmp_cells[ii + jj*nx].speeds[1] = cells[x_w + jj*nx].speeds[1]; /* east */
  // tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_s*nx].speeds[2]; /* north */
  // tmp_cells[ii + jj*nx].speeds[3] = cells[x_e + jj*nx].speeds[3]; /* west */
  // tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_n*nx].speeds[4]; /* south */
  // tmp_cells[ii + jj*nx].speeds[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
  // tmp_cells[ii + jj*nx].speeds[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
  // tmp_cells[ii + jj*nx].speeds[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
  // tmp_cells[ii + jj*nx].speeds[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */
}


kernel void rebound(global float* cells,
                    global float* tmp_cells,
                    global int* obstacles,
                    int nx, int ny)
{
  
  // /* get column and row indices */
  // int ii = get_global_id(0);
  // int jj = get_global_id(1);
  
  // /* if the cell contains an obstacle */
  // if (obstacles[jj*nx + ii])
  // {
  //   /* called after propagate, so taking values from scratch space
  //   ** mirroring, and writing into main grid */
  //   cells[ii + jj*nx].speeds[1] = tmp_cells[ii + jj*nx].speeds[3];
  //   cells[ii + jj*nx].speeds[2] = tmp_cells[ii + jj*nx].speeds[4];
  //   cells[ii + jj*nx].speeds[3] = tmp_cells[ii + jj*nx].speeds[1];
  //   cells[ii + jj*nx].speeds[4] = tmp_cells[ii + jj*nx].speeds[2];
  //   cells[ii + jj*nx].speeds[5] = tmp_cells[ii + jj*nx].speeds[7];
  //   cells[ii + jj*nx].speeds[6] = tmp_cells[ii + jj*nx].speeds[8];
  //   cells[ii + jj*nx].speeds[7] = tmp_cells[ii + jj*nx].speeds[5];
  //   cells[ii + jj*nx].speeds[8] = tmp_cells[ii + jj*nx].speeds[6];
  // }
}


kernel void collision(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      float omega,
                      int nx, int ny)
{
  // const float c_sq = 1.f / 3.f; /* square of speed of sound */
  // const float w0 = 4.f / 9.f;  /* weighting factor */
  // const float w1 = 1.f / 9.f;  /* weighting factor */
  // const float w2 = 1.f / 36.f; /* weighting factor */

  // /* get column and row indices */
  // int ii = get_global_id(0);
  // int jj = get_global_id(1);

  // /* don't consider occupied cells */
  // if (!obstacles[ii + jj*nx])
  // {
  //   /* compute local density total */
  //   float local_density = 0.f;

  //   for (int kk = 0; kk < NSPEEDS; kk++)
  //   {
  //     local_density += tmp_cells[ii + jj*nx].speeds[kk];
  //   }

  //   /* compute x velocity component */
  //   float u_x = (tmp_cells[ii + jj*nx].speeds[1]
  //                 + tmp_cells[ii + jj*nx].speeds[5]
  //                 + tmp_cells[ii + jj*nx].speeds[8]
  //                 - (tmp_cells[ii + jj*nx].speeds[3]
  //                     + tmp_cells[ii + jj*nx].speeds[6]
  //                     + tmp_cells[ii + jj*nx].speeds[7]))
  //                 / local_density;
  //   /* compute y velocity component */
  //   float u_y = (tmp_cells[ii + jj*nx].speeds[2]
  //                 + tmp_cells[ii + jj*nx].speeds[5]
  //                 + tmp_cells[ii + jj*nx].speeds[6]
  //                 - (tmp_cells[ii + jj*nx].speeds[4]
  //                     + tmp_cells[ii + jj*nx].speeds[7]
  //                     + tmp_cells[ii + jj*nx].speeds[8]))
  //                 / local_density;

  //   /* velocity squared */
  //   float u_sq = u_x * u_x + u_y * u_y;

  //   /* directional velocity components */
  //   float u[NSPEEDS];
  //   u[1] =   u_x;        /* east */
  //   u[2] =         u_y;  /* north */
  //   u[3] = - u_x;        /* west */
  //   u[4] =       - u_y;  /* south */
  //   u[5] =   u_x + u_y;  /* north-east */
  //   u[6] = - u_x + u_y;  /* north-west */
  //   u[7] = - u_x - u_y;  /* south-west */
  //   u[8] =   u_x - u_y;  /* south-east */

  //   /* equilibrium densities */
  //   float d_equ[NSPEEDS];
  //   /* zero velocity density: weight w0 */
  //   d_equ[0] = w0 * local_density
  //               * (1.f - u_sq / (2.f * c_sq));
  //   /* axis speeds: weight w1 */
  //   d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
  //                                     + (u[1] * u[1]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));
  //   d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
  //                                     + (u[2] * u[2]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));
  //   d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
  //                                     + (u[3] * u[3]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));
  //   d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
  //                                     + (u[4] * u[4]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));
  //   /* diagonal speeds: weight w2 */
  //   d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
  //                                     + (u[5] * u[5]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));
  //   d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
  //                                     + (u[6] * u[6]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));
  //   d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
  //                                     + (u[7] * u[7]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));
  //   d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
  //                                     + (u[8] * u[8]) / (2.f * c_sq * c_sq)
  //                                     - u_sq / (2.f * c_sq));

  //   /* relaxation step */
  //   for (int kk = 0; kk < NSPEEDS; kk++)
  //   {
  //     cells[ii + jj*nx].speeds[kk] = tmp_cells[ii + jj*nx].speeds[kk]
  //                                             + omega
  //                                             * (d_equ[kk] - tmp_cells[ii + jj*nx].speeds[kk]);
  //   }
  // }
}


kernel void process(global float* cells,
                    global float* tmp_cells,
                    global int* obstacles,
                    float omega,
                    int nx, int ny,
                    local  float*    local_tot_u,
                    local  int*    local_tot_cells,
                    global float*    partial_tot_u,
                    global int*    partial_tot_cells,
                    int timestep)
{
  //Propogate
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  int local_i = get_local_id(0);
  int local_j = get_local_id(1);
  int work_group_size = get_local_size(0);

  //av_vels variables
  int   tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  //Collision parameters
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  // tmp_cells[ii + jj*nx] = cells[(0 * ny * nx)ii + jj*nx]; /* central cell, no movement */
  // tmp_cells[(1 * ny * nx) + ii + jj*nx] = cells[(1 * ny * nx) + x_w + jj*nx]; /* east */
  // tmp_cells[(2 * ny * nx) + ii + jj*nx] = cells[(2 * ny * nx) + ii + y_s*nx]; /* north */
  // tmp_cells[(3 * ny * nx) + ii + jj*nx] = cells[(3 * ny * nx) + x_e + jj*nx]; /* west */
  // tmp_cells[(4 * ny * nx) + ii + jj*nx] = cells[(4 * ny * nx) + ii + y_n*nx]; /* south */
  // tmp_cells[(5 * ny * nx) + ii + jj*nx] = cells[(5 * ny * nx) + x_w + y_s*nx]; /* north-east */
  // tmp_cells[(6 * ny * nx) + ii + jj*nx] = cells[(6 * ny * nx) + x_e + y_s*nx]; /* north-west */
  // tmp_cells[(7 * ny * nx) + ii + jj*nx] = cells[(7 * ny * nx) + x_e + y_n*nx]; /* south-west */
  // tmp_cells[(8 * ny * nx) + ii + jj*nx] = cells[(8 * ny * nx) + x_w + y_n*nx]; /* south-east */

  const float tempSpeed0 = cells[ii + jj*nx]; /* central cell, no movement */
  const float tempSpeed1 = cells[(1 * ny * nx) + x_w + jj*nx]; /* east */
  const float tempSpeed2 = cells[(2 * ny * nx) + ii + y_s*nx]; /* north */
  const float tempSpeed3 = cells[(3 * ny * nx) + x_e + jj*nx]; /* west */
  const float tempSpeed4 = cells[(4 * ny * nx) + ii + y_n*nx]; /* south */
  const float tempSpeed5 = cells[(5 * ny * nx) + x_w + y_s*nx]; /* north-east */
  const float tempSpeed6 = cells[(6 * ny * nx) + x_e + y_s*nx]; /* north-west */
  const float tempSpeed7 = cells[(7 * ny * nx) + x_e + y_n*nx]; /* south-west */
  const float tempSpeed8 = cells[(8 * ny * nx) + x_w + y_n*nx]; /* south-east */

  //Rebound
  /* if the cell contains an obstacle */
  if (obstacles[jj*nx + ii])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    tmp_cells[(1 * ny * nx) + ii + jj*nx] = tempSpeed3;
    tmp_cells[(2 * ny * nx) + ii + jj*nx] = tempSpeed4;
    tmp_cells[(3 * ny * nx) + ii + jj*nx] = tempSpeed1;
    tmp_cells[(4 * ny * nx) + ii + jj*nx] = tempSpeed2;
    tmp_cells[(5 * ny * nx) + ii + jj*nx] = tempSpeed7;
    tmp_cells[(6 * ny * nx) + ii + jj*nx] = tempSpeed8;
    tmp_cells[(7 * ny * nx) + ii + jj*nx] = tempSpeed5;
    tmp_cells[(8 * ny * nx) + ii + jj*nx] = tempSpeed6;
  }

  //Collision
  else{
    /* compute local density total */
    float local_density = 0.f;

    local_density += tempSpeed0;
    local_density += tempSpeed1;
    local_density += tempSpeed2;
    local_density += tempSpeed3;
    local_density += tempSpeed4;
    local_density += tempSpeed5;
    local_density += tempSpeed6;
    local_density += tempSpeed7;
    local_density += tempSpeed8;

    /* compute x velocity component */
    float u_x = (tempSpeed1
                  + tempSpeed5
                  + tempSpeed8
                  - (tempSpeed3
                      + tempSpeed6
                      + tempSpeed7))
                  / local_density;
    /* compute y velocity component */
    float u_y = (tempSpeed2
                  + tempSpeed5
                  + tempSpeed6
                  - (tempSpeed4
                      + tempSpeed7
                      + tempSpeed8))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
    float u[NSPEEDS];
    u[1] =   u_x;        /* east */
    u[2] =         u_y;  /* north */
    u[3] = - u_x;        /* west */
    u[4] =       - u_y;  /* south */
    u[5] =   u_x + u_y;  /* north-east */
    u[6] = - u_x + u_y;  /* north-west */
    u[7] = - u_x - u_y;  /* south-west */
    u[8] =   u_x - u_y;  /* south-east */

    // Maths
    float var1 = 2.f * c_sq * c_sq;
    float var2 = u_sq / (2.f * c_sq);

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density
                * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                      + (u[1] * u[1]) / (var1)
                                      - var2);
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                      + (u[2] * u[2]) / (var1)
                                      - var2);
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                      + (u[3] * u[3]) / (var1)
                                      - var2);
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                      + (u[4] * u[4]) / (var1)
                                      - var2);
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                      + (u[5] * u[5]) / (var1)
                                      - var2);
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                      + (u[6] * u[6]) / (var1)
                                      - var2);
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                      + (u[7] * u[7]) / (var1)
                                      - var2);
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                      + (u[8] * u[8]) / (var1)
                                      - var2);

    /* relaxation step */
    tmp_cells[ii + jj*nx]                       = tempSpeed0
                                            + omega
                                            * (d_equ[0] - tempSpeed0);
    tmp_cells[(1 * ny * nx) + ii + jj*nx] = tempSpeed1
                                            + omega
                                            * (d_equ[1] - tempSpeed1);
    tmp_cells[(2 * ny * nx) + ii + jj*nx] = tempSpeed2
                                            + omega
                                            * (d_equ[2] - tempSpeed2);
    tmp_cells[(3 * ny * nx) + ii + jj*nx] = tempSpeed3
                                            + omega
                                            * (d_equ[3] - tempSpeed3);
    tmp_cells[(4 * ny * nx) + ii + jj*nx] = tempSpeed4
                                            + omega
                                            * (d_equ[4] - tempSpeed4);
    tmp_cells[(5 * ny * nx) + ii + jj*nx] = tempSpeed5
                                            + omega
                                            * (d_equ[5] - tempSpeed5);
    tmp_cells[(6 * ny * nx) + ii + jj*nx] = tempSpeed6
                                            + omega
                                            * (d_equ[6] - tempSpeed6);
    tmp_cells[(7 * ny * nx) + ii + jj*nx] = tempSpeed7
                                            + omega
                                            * (d_equ[7] - tempSpeed7);
    tmp_cells[(8 * ny * nx) + ii + jj*nx] = tempSpeed8
                                            + omega
                                            * (d_equ[8] - tempSpeed8);


    tot_u = sqrt((u_x * u_x) + (u_y * u_y));
  }

  int mask = (1 - obstacles[jj*nx + ii]);

  local_tot_u[local_i + local_j*work_group_size] = tot_u * mask;
  local_tot_cells[local_i + local_j*work_group_size] = mask;

  barrier(CLK_LOCAL_MEM_FENCE);
  reduce(local_tot_u, local_tot_cells, partial_tot_u, partial_tot_cells, timestep);
}


void reduce(local  float* local_tot_u,
            local  int* local_tot_cells,
            global float* partial_tot_u,
            global int* partial_tot_cells,
            int timestep)
{
  int group_idi = get_group_id(0);
  int group_idj = get_group_id(1);
  int nworkgroups1 = get_num_groups(0);
  int nworkgroups2 = get_num_groups(1);
  // printf(" kernel: %d ", nworkgroups);

  int local_i = get_local_id(0);
  int local_j = get_local_id(1);
  int work_group_size1 = get_local_size(0);
  int work_group_size2 = get_local_size(1);

  if(local_i == 0 && local_j == 0){
    //av_vels variables
    int   tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;

    for(int i = 0; i < work_group_size1*work_group_size2; i++){
      tot_u += local_tot_u[i];
      tot_cells += local_tot_cells[i];
      // printf(" (%d) ", local_tot_cells[i]);
    }
    int index = (timestep * nworkgroups1*nworkgroups2) + (group_idi + group_idj*nworkgroups1);
    // printf(" %d: ", index);
    partial_tot_u[index] = tot_u;
    partial_tot_cells[index] = tot_cells;
    // printf("   %f, %d  ", tot_u, tot_cells);
  }

  
  // printf(" (%d) ", tot_cells); 

}
