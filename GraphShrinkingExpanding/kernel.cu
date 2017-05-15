
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include <string.h>

#define SIZE_VERTICES 3
#define SIZE_EDGES 5
#define DEBUG_MODE false

void load_graph_from_edge_list_file(int*, int*, char*);
void print_debug(char*);
void print_debug(char*, int);
void print_edge_list(int*, int*);

int main() {
	int* source_vertices;
	int* target_vertices;
	char* file_path = "C:\\Users\\AJ\\Documents\\example_graph.txt";

	source_vertices = (int*) malloc(sizeof(int) * SIZE_EDGES);
	target_vertices = (int*) malloc(sizeof(int) * SIZE_EDGES);

	load_graph_from_edge_list_file(source_vertices, target_vertices, file_path);
	
	print_edge_list(source_vertices, target_vertices);

	// Cleanup
	free(source_vertices);
	free(target_vertices);

	return 0;
}

/*
NOTE: Only reads integer vertices for now (through the 'sscanf' function) and obvious input vertices arrays
*/
void load_graph_from_edge_list_file(int* source_vertices, int* target_vertices, char* file_path) {
	printf("Loading graph file: %s", file_path);

	FILE* file = fopen(file_path, "r");
	char line[256];

	int edge_index = 0;

	while (fgets(line, sizeof(line), file)) {
		if (line[0]=='#') {
			print_debug("\nEscaped a comment.");
			continue;
		}

		// Save source and target vertex (temp)
		int source_vertex;
		int target_vertex;

		sscanf(line, "%d%d\t%[^\n]", &source_vertex, &target_vertex);

		// Add vertices to the source and target arrays, forming an edge accordingly
		source_vertices[edge_index] = source_vertex;
		target_vertices[edge_index] = target_vertex;

		// Increment edge index to add any new edge
		edge_index++;
		
		// Debug: Print source and target vertex
		print_debug("\nAdded start vertex:", source_vertex);
		print_debug("\nAdded end vertex:", target_vertex);
	}
	
	fclose(file);
}

void print_edge_list(int* source_vertices, int* end_vertices) {
	for (int i = 0; i < SIZE_EDGES; i++) {
		printf("\n%d, %d", source_vertices[i], end_vertices[i]);
	}
}

void print_debug(char* message) {
	if (DEBUG_MODE)
		printf("%s", message);
}

void print_debug(char* message, int value) {
	if (DEBUG_MODE)
		printf("%s %d", message, value);
}