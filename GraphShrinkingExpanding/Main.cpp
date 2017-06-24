#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "sampling/Sampling.h"

const int MINIMUM_REQUIRED_INPUT_PARAMETERS = 4;

int main(int argc, char* argv[]) {
	if (argc >= MINIMUM_REQUIRED_INPUT_PARAMETERS) {
		char* input_path = argv[1];
		char* output_path = argv[2];

		if (strcmp(argv[3], "sample") == 0) {
			Sampling* sampler = new Sampling();
			sampler->collect_sampling_parameters(argv);
			sampler->sample_graph(input_path, output_path);
		}
		else {
			//collect_expanding_parameters(argv);
			//expand_graph(input_path, output_path, EXPANDING_FACTOR);
		}
	} 
	else {
		printf("Incorrect amount of input/output arguments given.");

		// ONLY FOR LOCAL TESTING
		//char* input_path = "C:\\Users\\AJ\\Documents\\example_graph.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\nvgraphtest\\nvGraphExample-master\\nvGraphExample\\web-Stanford_large.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\edge_list_example.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\roadnet.txt";
		char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\facebook_graph.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\output_test\\social\\soc-pokec-relationships.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\roadNet-PA.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\soc-pokec-relationships.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\com-orkut.ungraph.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\soc-LiveJournal1.txt";
		//char* input_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\coo\\pokec_coo.txt";
		char* output_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\output\\fb_sampled_refactored.txt";

		Sampling* sampler = new Sampling();
		sampler->SAMPLING_FRACTION = 0.5;
		sampler->sample_graph(input_path, output_path);
		/*
		EXPANDING_FACTOR = 3;
		SAMPLING_FRACTION = 0.5;
		SELECTED_TOPOLOGY = STAR;
		SELECTED_BRIDGE_NODE_SELECTION = RANDOM_NODES;
		AMOUNT_INTERCONNECTIONS = 10;
		FORCE_UNDIRECTED_BRIDGES = true;
		expand_graph(input_path, output_path, EXPANDING_FACTOR);*/
	}

	return 0;
}