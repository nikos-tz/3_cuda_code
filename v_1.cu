//%%cu

/*** V_1 ***/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


__global__ void v_1(int* read, int* write, int n);

int main() {

    // test it for several n and their corresponding k

    int nV[] = { 1024, 4000, 8000, 12000, 16000 };
    int kV[] = { 10, 28, 30, 32, 35 };

    double myTime = 0.0; // for the results
    struct timeval start, end;

    int threads_per_block_sqrt = 32; // max because 32*32=1024


    for(int index=0; index < 5; ++index) {

        int n = nV[index];
        int k = kV[index];
        int size = n * n;
        int num_blocks_sqrt = n / threads_per_block_sqrt;

        int num_blocks = num_blocks_sqrt * num_blocks_sqrt;
        int threads_per_block = threads_per_block_sqrt * threads_per_block_sqrt;

        printf("\nV1: n=%d, k=%d, [blocks=%d, threads/block=%d]\n\n", n, k, num_blocks, threads_per_block);

        //host variables
        int *h_lattice_1 = (int *) calloc(size, sizeof(int));
        int *h_lattice_2 = (int *) calloc(size, sizeof(int));
        //result
        int *lattice_k = (int *) calloc(size, sizeof(int));
        //device variables
        int *d_read;
        int *d_write;

        const int d_lattice_size = size * sizeof(int);

        // initialize memory for them

        cudaMalloc((void **) &d_read, d_lattice_size);
        cudaMalloc((void **) &d_write, d_lattice_size);


        srand((unsigned int) time(NULL));

        // for every n do it 10 times

        for(int iterations=0; iterations < 10; ++iterations) {

            for (int i = 0; i < size; ++i) {
                int value = (int) rand() % 2;

                h_lattice_1[i] = value ? 1 : -1;
            }

            // give them values

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


            int *temp = NULL; // it is going to point to device memory

            /*** CALCUALTE ***/



            gettimeofday(&start, NULL); //Start timing the computation

            // call kernel k times

            for (int i = 0; i < k; ++i) {

                v_1<<< num_blocks, threads_per_block >>>(d_read, d_write, n);

                cudaDeviceSynchronize();

                temp = d_write; // temp always has the result
                d_write = d_read;
                d_read = temp;

            }

            gettimeofday(&end, NULL); //Stop timing the computation

            myTime = (end.tv_sec + (double) end.tv_usec / 1000000) - (start.tv_sec + (double) start.tv_usec / 1000000);

            // copy the result from device back to the host
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


    return 0;
}

__global__ void v_1(int* read, int* write, int n) {

    // make all the indexing

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("hello from thread: %d\n", thread_id);

    int i = thread_id / n;
    int j = thread_id % n;

    // same process as v_0

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

    if(thread_id == 0){
        // printf("value is %d from thread: %d\n", write[i*n + j], thread_id);
    }


}



