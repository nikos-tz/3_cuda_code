// %%cu

/*** V_2 ***/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


__global__ void v_2(int* read, int* write, int n, int moments_per_thread_sqrt);

int main() {

    int nV[] = { 1024, 4000, 8000, 12000, 16000 };
    int kV[] = { 10, 28, 30, 32, 35 };
    int bV[] = { 2, 3, 4, 5, 6, 7, 8};

    double myTime = 0.0;
    struct timeval start,end;

    int threads_per_block_sqrt = 32;

    for(int index=0; index < 5; ++index) {

       for(int b_index=0; b_index < 7; ++b_index) {

            int n = nV[index];
            int moments_per_thread_sqrt = bV[b_index];
            int num_blocks_sqrt = n / (threads_per_block_sqrt * moments_per_thread_sqrt);
            int size = n * n;
            int k = kV[index];

            printf("\nV2: n=%d, k=%d, [blocks=%d, threads/block=%d, moments/thread=%d, all squared]\n\n",
                   n, k, num_blocks_sqrt, threads_per_block_sqrt, moments_per_thread_sqrt);

            int *h_lattice_1 = (int *) calloc(size, sizeof(int));
            int *h_lattice_2 = (int *) calloc(size, sizeof(int));
            int *lattice_k = (int *) calloc(size, sizeof(int));

           int *d_read;
           int *d_write;

           const int d_lattice_size = size * sizeof(int);

           cudaMalloc((void **) &d_read, d_lattice_size);
           cudaMalloc((void **) &d_write, d_lattice_size);


           srand((unsigned int) time(NULL));

           for(int iterations=0; iterations < 10; ++iterations) {

               for (int i = 0; i < size; ++i) {
                   int value = (int) rand() % 2;

                   h_lattice_1[i] = value ? 1 : -1;
               }

               cudaMemcpy(d_read, h_lattice_1, d_lattice_size, cudaMemcpyHostToDevice);
               cudaMemcpy(d_write, h_lattice_2, d_lattice_size, cudaMemcpyHostToDevice);

               /*** print ***/
               /*
               for(int i=0; i < n; ++i){
                   for(int j=0; j < n; ++j){
                       printf("%d\t", h_lattice_1[i*n + j]);
                   }
                   printf("\n");
               }
               */


               int *temp = NULL;

               /*** CALCUALTE ***/

               dim3 dimBlock(threads_per_block_sqrt, threads_per_block_sqrt);
               dim3 dimGrid(num_blocks_sqrt, num_blocks_sqrt);


               gettimeofday(&start, NULL); //Start timing the computation

               for (int i = 0; i < k; ++i) {

                   v_2<<< dimGrid, dimBlock >>>(d_read, d_write, n, moments_per_thread_sqrt);

                   cudaDeviceSynchronize();

                   temp = d_write;
                   d_write = d_read;
                   d_read = temp;

               }

               gettimeofday(&end, NULL); //Stop timing the computation

               myTime = (end.tv_sec + (double) end.tv_usec / 1000000) -
                        (start.tv_sec + (double) start.tv_usec / 1000000);

               cudaMemcpy(lattice_k, temp, d_lattice_size, cudaMemcpyDeviceToHost);


               /*** print ***/

               printf("%lf\n", myTime);
               /*
               printf("After %d generations:\n", k);

               for(int i=0; i < n; ++i){
                   for(int j=0; j < n; ++j){
                       printf("%d\t", lattice_k[i*n + j]);
                   }
                   printf("\n");
               }
               */


           }

            cudaFree(d_write);
            cudaFree(d_read);


            free(h_lattice_1);
            free(h_lattice_2);
            free(lattice_k);

        }
    }

    return 0;
}

__global__ void v_2(int* read, int* write, int n, int moments_per_thread_sqrt) {

    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    int thread_i = thread_id_y * moments_per_thread_sqrt;
    int thread_j = thread_id_x * moments_per_thread_sqrt;

    for(int i = thread_i; i < thread_i + moments_per_thread_sqrt; ++i) {
        for(int j = thread_j; j < thread_j + moments_per_thread_sqrt; ++j) {

            int i_minus_one = (n + ((i - 1) % n)) % n;
            int i_plus_one = (n + ((i + 1) % n)) % n;
            int j_minus_one = (n + ((j - 1) % n)) % n;
            int j_plus_one = (n + ((j + 1) % n)) % n;

            int value = (read[i * n + j]
                         + read[i_minus_one * n + j]
                         + read[i_plus_one * n + j]
                         + read[i * n + j_minus_one]
                         + read[i * n + j_plus_one]);

            write[i * n + j] = (value > 0) ? 1 : -1;

        }
    }



}