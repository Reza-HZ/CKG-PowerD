# CKG-PowerD
Identifying key genes in cancer based on power dominating set


Manual for CKG-PowerD

Installation
Before using the code, make sure you have Python installed on your machine. The code requires the following libraries, which can be installed using pip:


pip install networkx pandas numpy

Usage
To run the code, open a terminal or command prompt and navigate to the directory where the code is located. Use the following command:


python script.py -i input_graph.graphml -s PDS_SIZE -o output_file.txt -p is_parallel

Replace script.py with the actual filename of the code, and the following parameters:

-i or --Input_filename: Path to the input graph file in GraphML format.
-s or --pds_size: The size of the PDS sets to find.
-o or --Output_filename: (Optional) The desired filename for the output file. If not provided, the default filename will be used.
-p or --is_parallel: (Optional) A flag to indicate whether to use multiprocessing (1) or not (0) when finding PDS sets. If not provided, the default value is 0 (no multiprocessing).

Example
Suppose you have a graph file named my_graph.graphml, and you want to find all PDS sets of size 3 using parallel processing. You also have a text file named my_text.txt. To achieve this, run the following command:

python script.py -i my_graph.graphml -s 3 -o output_results.txt -p 1
