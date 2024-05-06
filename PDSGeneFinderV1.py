from itertools import combinations
import getopt, sys
import numpy as np
import multiprocessing as mp
import networkx as nx
import pandas as pd
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)

argumentList = sys.argv[1:]
options = "i:s:o:p:"
long_options = ["Input_filename=", "pds_size=", "Output_filename=", "is_parallel="]
output_Filename = ""
filename = ""
pds_size = None
is_parallel = None
try:
    arguments, values = getopt.getopt(argumentList, options, long_options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-i", "--Input_filename"):
            filename = currentValue
        elif currentArgument in ("-s", "--pds_size"):
            pds_size = int(currentValue)
        elif currentArgument in ("-o", "--Output_filename"):
            output_Filename = currentValue
        elif currentArgument in ("-p", "--is_parallel"):
            is_parallel = int(currentValue)
except getopt.error as err:
    print (str(err))
if (pds_size == None):
    pds_size = 1
if (is_parallel == None):
    is_parallel = 0
if not output_Filename:
    output_Filename = filename + '_' +str(pds_size) +'_results.txt'

output_Filename_counts = 'counts_' + output_Filename

def generate_combinations(n, k):
    elements = range(0, n)
    yield from combinations(elements, k)

grphrawdata = nx.read_graphml(filename)
graphdata = grphrawdata.to_undirected()
A = nx.adjacency_matrix(graphdata).toarray()
nodes_names_df = pd.DataFrame({'num': range(len(graphdata.nodes())),'name': list(graphdata.nodes())})
all_testsets = generate_combinations(len(A), pds_size)

def calc_allpds():
    PDS_Matrix = np.empty((0, pds_size), int)
    result = map(is_pds, all_testsets)
    non_none_list = [element for element in result if element is not None]
    for entry in non_none_list:
        PDS_Matrix = np.vstack([PDS_Matrix, [graphdata.nodes[x]['name'] for x in entry]])
    return PDS_Matrix

def is_pds(testsettemp):
    testset = [list(graphdata.nodes)[i] for i in testsettemp]
    flag = True
    nodes_num = len(graphdata.nodes)
    tps = np.unique(np.concatenate([np.array(list(graphdata.neighbors(v))) for v in testset] + [testset]))
    tps_vec = np.zeros(nodes_num, int)
    index = nodes_names_df[nodes_names_df["name"].isin(tps)].index
    tps_vec[index] = 1
    while True:
        if len(tps) == nodes_num:
            break
        tempmat = A.copy()
        tempmat[:, index] *= (1 - tps_vec)[:, np.newaxis]
        inds = np.where(tempmat.sum(axis=0) == 1)[0]
        del tempmat
        inds = np.setdiff1d(np.intersect1d(inds, index), testset)
        if len(inds) == 0:
            flag = False
            break
        else:
            tps = np.unique(np.concatenate([tps, np.concatenate([list(graphdata.neighbors(v)) for v in nodes_names_df.name[inds]])]))
            tps_vec = np.zeros(nodes_num, int)
            index = nodes_names_df[nodes_names_df["name"].isin(tps)].index
            tps_vec[index] = 1
    if (flag):
        return testset

def count_word_occurrences(input_file, output_file, pds_size):
    with open(input_file, 'r') as file:
        text = file.read()
    words = text.split()
    n = len(words)/pds_size
    word_counts = Counter(words)
    sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    with open(output_file, 'w') as file:
        for word, count in sorted_counts:
            file.write(f"{word}: {(count/n)*100}\n")

if __name__ == "__main__":
    mp.freeze_support()
    if (is_parallel == 1):
        #num_processes = mp.cpu_count()
        pool = mp.Pool(4)
        temp_results = [result for result in pool.map(is_pds, all_testsets) if result]
        pool.close()
        pool.join()
        PDS_Matrix = np.empty((0, pds_size), str)
        if len(temp_results):
            PDS_Matrix = np.vstack([PDS_Matrix, [[graphdata.nodes[x]['name'] for x in i] for i in temp_results]])
    else:
        PDS_Matrix = calc_allpds()
    if (np.size(PDS_Matrix) != 0):
        np.savetxt(output_Filename, PDS_Matrix, fmt="%s")
        count_word_occurrences(output_Filename, output_Filename_counts, pds_size)
        print('The process is finished and the results were recorded in the output file.')
    else:
        print('There is no PDS!')
