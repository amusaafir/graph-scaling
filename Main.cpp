#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <mpi.h>
#include "sampling/Sampling.h"


int main(int argc, char* argv[]) {
	int mpi_id;
	int mpi_size;

	/* Initialize MPI. */
	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank (MPI_COMM_WORLD, &mpi_id);

	if (mpi_id == 0) {
		printf("Running with %d MPI processes.\n", mpi_size);
	}

	const int MINIMUM_REQUIRED_INPUT_PARAMETERS = 6;

	if (argc >= MINIMUM_REQUIRED_INPUT_PARAMETERS) {
		char* input_path = argv[1];
		char* output_path = argv[2];

		GraphIO* graph_io = new GraphIO();
		graph_io->collect_sampling_parameters(argv);
		Sampling* sampler = new Sampling(graph_io);

		sampler->collect_sampling_parameters(argv);
		sampler->sample_graph(input_path, output_path, mpi_size, mpi_id);

		delete(sampler);
		delete(graph_io);

	} else {
		if (mpi_id == 0) {
			printf("Incorrect amount of input/output arguments given.\n");
			printf("Usage: ./sample <input file> <output file> ");
			printf("<sampling fraction> <CPU memory size (bytes)> ");
			printf("<GPU memory size (bytes)>\n");
		}
	}

	MPI_Finalize();

	if (mpi_id == 0)
		printf("\nExecution terminates.\n");

	return 0;
}
