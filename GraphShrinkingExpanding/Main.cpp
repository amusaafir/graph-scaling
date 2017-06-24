#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "expanding/Expanding.h"
#include "sampling/Sampling.h"

int main(int argc, char* argv[]) {
	const int MINIMUM_REQUIRED_INPUT_PARAMETERS = 4;

	if (argc >= MINIMUM_REQUIRED_INPUT_PARAMETERS) {
		char* input_path = argv[1];
		char* output_path = argv[2];
		GraphIO* graph_io = new GraphIO();

		if (strcmp(argv[3], "sample") == 0) {
			Sampling* sampler = new Sampling(graph_io);
			sampler->collect_sampling_parameters(argv);
			sampler->sample_graph(input_path, output_path);
			delete(sampler);
		}
		else {
			Expanding* expander = new Expanding(graph_io);
			expander->collect_expanding_parameters(argv);
			expander->expand_graph(input_path, output_path);
			delete(expander);
		}

		delete(graph_io);
	} else {
		printf("Incorrect amount of input/output arguments given.");

		// THE FOLLOWING CODE IS ONLY FOR LOCAL TESTING

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
		char* output_path = "C:\\Users\\AJ\\Desktop\\new_datasets\\output\\fb_expanded.txt";

		GraphIO* graph_io = new GraphIO();

		/*Sampling* sampler = new Sampling(graph_io);
		sampler->SAMPLING_FRACTION = 0.5;
		sampler->sample_graph(input_path, output_path);*/
		
		Expanding* expander = new Expanding(graph_io);
		expander->SCALING_FACTOR = 3;
		expander->SAMPLING_FRACTION = 0.5;
		expander->set_topology(new Star(10, new RandomBridge(), true));
		expander->SELECTED_BRIDGE_NODE_SELECTION = RANDOM_NODES;
		//expander->AMOUNT_INTERCONNECTIONS = 10;
		expander->FORCE_UNDIRECTED_BRIDGES = true;
		expander->expand_graph(input_path, output_path);

		delete(expander);
		delete(graph_io);
	}

	return 0;
}