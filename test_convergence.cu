#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/** its the v_1 but for every k it computes the state
 *  at k+2 to see if they are the same state
 */


__global__ void v_1(int* read, int* write, int n);

int main() {

    int num_blocks_sqrt = 500;
    int threads_per_block_sqrt = 32;

    int n = num_blocks_sqrt * threads_per_block_sqrt;
    int k = 1;
    int size = n*n;

    int num_blocks = num_blocks_sqrt * num_blocks_sqrt;
    int threads_per_block = threads_per_block_sqrt * threads_per_block_sqrt;

    printf("\nV1: n=%d, k=%d, [blocks=%d, threads/block=%d]\n\n", n, k, num_blocks, threads_per_block);


    int* h_lattice_1 = (int *) calloc( size, sizeof(int));
    int* h_lattice_2 = (int *) calloc( size, sizeof(int));
    int* lattice_k = (int *) calloc( size, sizeof(int));
    int* lattice_k_plus_2 = (int *) calloc( size, sizeof(int)); // for the k+2 state

    srand((unsigned int)time(NULL));

    //do it a bunch of times because for the same n we might have different k

    for(int iterations=0; iterations < 5; ++iterations) {

        for (int i = 0; i < size; ++i) {
            int value = (int) rand() % 2;

            h_lattice_1[i] = value ? 1 : -1;
        }


        /*** print ***/
        /*
        for(int i=0; i < n; ++i){
            for(int j=0; j < n; ++j){
                printf("%d\t", h_lattice_1[i*n + j]);
            }
            printf("\n");
        }
        */

        // do it for some k
        for(k=28; k < 36; ++k) {

            int *d_read;
            int *d_write;

            const int d_lattice_size = size * sizeof(int);

            cudaMalloc((void **) &d_read, d_lattice_size);
            cudaMalloc((void **) &d_write, d_lattice_size);

            cudaMemcpy(d_read, h_lattice_1, d_lattice_size, cudaMemcpyHostToDevice);
            cudaMemcpy(d_write, h_lattice_2, d_lattice_size, cudaMemcpyHostToDevice);


            int *temp = NULL;

            /*** CALCUALTE ***/

            double myTime = 0.0;
            struct timeval start, end;

            gettimeofday(&start, NULL); //Start timing the computation

            for (int i = 0; i < k; ++i) {

                v_1<<< num_blocks, threads_per_block >>>(d_read, d_write, n);

                cudaDeviceSynchronize();

                temp = d_write;
                d_write = d_read;
                d_read = temp;

            }

            gettimeofday(&end, NULL); //Stop timing the computation

            myTime = (end.tv_sec + (double) end.tv_usec / 1000000) - (start.tv_sec + (double) start.tv_usec / 1000000);

            cudaMemcpy(lattice_k, temp, d_lattice_size, cudaMemcpyDeviceToHost);

            // now compute the k+2 state

            for (int i = 0; i < 2; ++i) {

                v_1<<< num_blocks, threads_per_block >>>(d_read, d_write, n);

                cudaDeviceSynchronize();

                temp = d_write;
                d_write = d_read;
                d_read = temp;

            }

            cudaMemcpy(lattice_k_plus_2, temp, d_lattice_size, cudaMemcpyDeviceToHost);

            /*** print ***/

            //printf("time: %lf\n\n", myTime);
            /*
            printf("After %d generations:\n", k);

            for(int i=0; i < n; ++i){
                for(int j=0; j < n; ++j){
                    printf("%d\t", lattice_k[i*n + j]);
                }
                printf("\n");
            }

             printf("\n");

            for(int i=0; i < n; ++i){
                for(int j=0; j < n; ++j){
                    printf("%d\t", lattice_k_plus_2[i*n + j]);
                }
                printf("\n");
            }
            */

            // it counts how many values of the k and the k+2 state are not the same

            int error_counter = 0;

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (lattice_k[i * n + j] != lattice_k_plus_2[i * n + j])
                        ++error_counter;
                }
            }


            // if they are the same then got our right k, so brake
            if( error_counter == 0) {
                printf("%d\n", k);
                break;
            }


            //printf("\nthere are %d errors\n", error_counter);


            cudaFree(d_write);
            cudaFree(d_read);

        }
    }

    free(h_lattice_1);
    free(h_lattice_2);
    free(lattice_k);


    return 0;
}

__global__ void v_1(int* read, int* write, int n) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;



    int i = thread_id / n;
    int j = thread_id % n;

    int i_minus_one = ( n + ( (i-1) % n ) ) % n;
    int i_plus_one = ( n + ( (i+1) % n ) ) % n;
    int j_minus_one = ( n + ( (j-1) % n ) ) % n;
    int j_plus_one = ( n + ( (j+1) % n ) ) % n;

    int value = ( read[i*n + j]
                  + read[i_minus_one*n + j]
                  + read[i_plus_one*n + j]
                  + read[i*n + j_minus_one]
                  + read[i*n + j_plus_one] );

    write[i*n + j] = (value > 0) ? 1 : -1;



}