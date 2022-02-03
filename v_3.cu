//%%cu

/*** V_3 ***/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


__global__ void v_3(int* read, int* write, int n, int threads_per_block_sqrt, int moments_per_thread_sqrt);

int main() {

    // same as v_2, but we are using shared memory

    int nV[] = { 1024, 4000, 8000, 12000, 16000 };
    int kV[] = { 10, 28, 30, 32, 35 };
    int bV[] = { 2, 4, 8 };

    double myTime = 0.0;
    struct timeval start,end;


    for(int index=0; index < 5; ++index) {

        for(int b_index=0; b_index < 3; ++b_index) {

            int n = nV[index];
            int moments_per_thread_sqrt = bV[b_index];
            int threads_per_block_sqrt = 32 / moments_per_thread_sqrt;
            int num_blocks_sqrt = n / (threads_per_block_sqrt * moments_per_thread_sqrt);
            int size = n * n;
            int k = kV[index];

            printf("\nV3: n=%d, k=%d, [blocks=%d, threads/block=%d, moments/thread=%d, all squared]\n\n",
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

                // dimension of the 2D shared memory array

                int shared_dim = (threads_per_block_sqrt * moments_per_thread_sqrt) + 2;


                gettimeofday(&start, NULL); //Start timing the computation

                for (int i = 0; i < k; ++i) {

                    /** the 3rd value into the <<< >>> is the shared memory we will
                     * use in bytes it is *2 because we will use 2 of them
                     */

                    v_3<<< dimGrid, dimBlock, 2 * shared_dim*shared_dim * sizeof(int) >>>
                        (d_read, d_write, n, threads_per_block_sqrt, moments_per_thread_sqrt);

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

__global__ void v_3(int* read, int* write, int n, int threads_per_block_sqrt, int moments_per_thread_sqrt) {



    int shared_dim = (threads_per_block_sqrt * moments_per_thread_sqrt) + 2;

    // extern because its size is not known at compile time
    extern __shared__ int shared[];
    // split the memory in two arrays
    int* read_shared = shared; // it is looking at the first one
    int* write_shared = (int*) &shared[shared_dim * shared_dim]; // it is looking at the second one

    // indexing both the global and the shared arrays

    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    int thread_i = thread_id_y * moments_per_thread_sqrt;
    int thread_j = thread_id_x * moments_per_thread_sqrt;

    int shared_i = threadIdx.y * moments_per_thread_sqrt;
    int shared_j = threadIdx.x * moments_per_thread_sqrt;

    /** now give values for the outer part of the shared array
     *  this part is going to be computed from the other blocks
     *  4 if statements because our square matrix has 4 sides
     *  and each if gets the values from the neighbors from one side
     */

    if( threadIdx.y == 0) {

        int i_minus_one = (n + ((thread_i - 1) % n)) % n;
        int j_host = thread_j;

        for(int j=shared_j; j < shared_j + moments_per_thread_sqrt; ++j) {
            read_shared[j+1] = read[i_minus_one * n + j_host];
            ++j_host;
        }
    }
    if(threadIdx.y == threads_per_block_sqrt - 1) {

        int i_plus_one = (n + ((thread_i + moments_per_thread_sqrt) % n)) % n;
        int j_host = thread_j;

        for(int j=shared_j; j < shared_j + moments_per_thread_sqrt; ++j) {
            read_shared[shared_dim * (shared_dim-1) + j + 1] = read[i_plus_one * n + j_host];
            //printf("%d (%d,%d)\n", read_shared[shared_dim * (shared_dim-1) + j + 1], (shared_dim * (shared_dim-1) + j + 1)/shared_dim, (shared_dim * (shared_dim-1) + j + 1)%shared_dim);
            ++j_host;
        }

    }
    if(threadIdx.x == 0) {

        int j_minus_one = (n + ((thread_j - 1) % n)) % n;
        int i_host = thread_i;

        for(int i=shared_i; i < shared_i + moments_per_thread_sqrt; ++i) {
            read_shared[(i+1)*shared_dim] = read[i_host * n + j_minus_one];
            //printf("%d (%d,%d)\n", read_shared[(i+1) * shared_dim], ((i+1)*shared_dim)/shared_dim, ((i+1)*shared_dim)%shared_dim);
            ++i_host;
        }

    }
    if(threadIdx.x == threads_per_block_sqrt - 1) {

        int j_plus_one = (n + ((thread_j + moments_per_thread_sqrt) % n)) % n;
        int i_host = thread_i;

        for(int i=shared_i; i < shared_i + moments_per_thread_sqrt; ++i) {
            read_shared[(i+1)*shared_dim + shared_dim - 1] = read[i_host * n + j_plus_one];
            //printf("%d (%d,%d)\n", read_shared[(i+1)*shared_dim + shared_dim - 1], ((i+1)*shared_dim + shared_dim - 1)/shared_dim, ((i+1)*shared_dim + shared_dim - 1)%shared_dim);
            ++i_host;
        }
    }

    int i_host = thread_i;

    /** now get the values for the inner part of the shared matrix
     *  this part will be computed from this block
     */


    for(int i = (shared_i + 1); i < (shared_i + 1 + moments_per_thread_sqrt); ++i) {

        int j_host = thread_j;

        for(int j = (shared_j + 1); j < (shared_j + 1 + moments_per_thread_sqrt); ++j) {
            read_shared[i * shared_dim + j] = read[i_host * n + j_host];
            //printf("%d (%d,%d)\n", read_shared[i * shared_dim + j], i, j);
            ++j_host;
        }

        ++i_host;
    }

    __syncthreads(); // synchronize all threads

    /** now compute each thread computes its values
     *  note that we dont need to check for periodic
     *  boundary conditions because we already did that
     *  when whe got the outer part of the matrix
     */


    for(int i = (shared_i + 1); i < (shared_i + 1 + moments_per_thread_sqrt); ++i) {
        for(int j = (shared_j + 1); j < (shared_j + 1 + moments_per_thread_sqrt); ++j) {

            int value = (read_shared[i * shared_dim + j]
                         + read_shared[(i-1) * shared_dim + j]
                         + read_shared[(i+1) * shared_dim + j]
                         + read_shared[i * shared_dim + (j-1)]
                         + read_shared[i * shared_dim + (j+1)]);

            write_shared[i * shared_dim + j] = (value > 0) ? 1 : -1;

        }
    }


    i_host = thread_i;

    // now put the values from shared to global memory

    for(int i = (shared_i + 1); i < (shared_i + 1 + moments_per_thread_sqrt); ++i) {

        int j_host = thread_j;

        for(int j = (shared_j + 1); j < (shared_j + 1 + moments_per_thread_sqrt); ++j) {
            write[i_host * n + j_host] = write_shared[i * shared_dim + j];
            ++j_host;
        }

        ++i_host;
    }




}