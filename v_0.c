#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

/*** V_0 ***/

int* v_0(int* read, int* write, int n, int k);

int main(int argc, char** argv) {

    int n = 1024;
    int k = 35;
    int size = n*n;

    FILE *file;
    file = fopen("/home/csal/pds/pds-codebase/ergasia/3rd exercise/files/v_0_16000.txt", "w");

    double myTime = 0.0; // for the results
    struct timeval start, end;

    // two lattices, one for read and one for write
    int* lattice_1;
    lattice_1 = (int *) calloc( size, sizeof(int));
    int* lattice_2;
    lattice_2 = (int *) calloc( size, sizeof(int));

    srand((unsigned int)time(NULL));


        // set random values

        for (int i = 0; i < size; ++i) {
            int value = (int) rand() % 2;

            lattice_1[i] = value ? 1 : -1; // +1 for positive spin and -1 for negative one
        }



        /*** print ***/
        /*
        for(int i=0; i < n; ++i){
            for(int j=0; j < n; ++j){
                printf("%d\t", lattice_1[i*n + j]);
            }
            printf("\n");
        }
        */


        gettimeofday(&start, NULL);

        int *lattice_generation_k = v_0(lattice_1, lattice_2, n, k); //lattice after k states

        gettimeofday(&end, NULL);

        myTime = (end.tv_sec + (double) end.tv_usec / 1000000) - (start.tv_sec + (double) start.tv_usec / 1000000);

        //fprintf(file, "%lf\n", myTime);

        /*** print ***/
        /*
        printf("After %d generations:\n", k);

        for(int i=0; i < n; ++i){
            for(int j=0; j < n; ++j){
                printf("%d\t", lattice_generation_k[i*n + j]);
            }
            printf("\n");
        }
        */



    fclose(file);


    free(lattice_1);
    free(lattice_2);

    return (0);
}

int* v_0(int* read, int* write, int n, int k) {

    int* temp = NULL;

    for(int l=0; l < k; ++l){

        //printf("Generation %d:\n", l);

        for(int i=0; i < n; ++i){
            for(int j=0; j < n; ++j){

                // see the neighbors with periodic boundary conditions

                int i_minus_one = ( n + ( (i-1) % n ) ) % n;
                int i_plus_one = ( n + ( (i+1) % n ) ) % n;
                int j_minus_one = ( n + ( (j-1) % n ) ) % n;
                int j_plus_one = ( n + ( (j+1) % n ) ) % n;

                // compute the value

                int value = ( read[i*n + j]
                            + read[i_minus_one*n + j]
                            + read[i_plus_one*n + j]
                            + read[i*n + j_minus_one]
                            + read[i*n + j_plus_one] );

                write[i*n + j] = (value > 0) ? 1 : -1; // decide if it is positive or negative

                //printf("%d\t", write[i*n + j]);

            }
            //printf("\n");
        }

        /* temp always has our answer*/
        temp = write;

        //swipe the pointers

        write = read;
        read = temp;

    }

    return temp;

}
