#%% Create a r100 testset for mikenet tf model that match with train_r100
import data_wrangling
import numpy as np
import os

os.chdir('/home/jupyter/triangle_model/')

#%% Parser class

class MikeNetPattern:
    """MikeNet pattern (weight) parser"""
    ort_units = 364
    pho_units = 200
    sem_units = 2446

    def __init__(self, file:str = None):

        if file is None:
            file = "mikenet/englishdict_randcon.pat.txt"

        self.items = list()
        self.ort_pattern = {}
        self.pho_pattern = {}
        self.sparse_sem_pattern = {}

        with open(file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("TAG Word: "):
                    content = line.split(" ")
                    word = content[2]
                    ort = content[6] # human readable ort
                    pho = content[4] # human readable pho
                    print(f"word:{word}, ort:{ort}, pho:{pho}")

                    # Record item
                    self.items.append(word)

                if line.startswith("CLAMP Ortho"):
                    # Take the next 14 lines value into ort representation dictionary
                    self.ort_pattern[word] = self.get_pattern(lines, i+1, i+15)

                if line.startswith("TARGET Phono"):
                    # Take the next 8 lines value into pho representation dictionary
                    self.pho_pattern[word] = self.get_pattern(lines, i+1, i+9)

                if line.startswith("TARGET Semantics"):
                    self.sparse_sem_pattern[word] = self.get_pattern(lines, i+1, i+2)

    def save(self):
        """Save r100 testset to file mn_r100.pkl.gz

        # Pack into my TF data format
        # Important: the order of the items must be the same as the order of the inputs
        # I did not account for multi meaning items
        """

        mn_r100 = {}

        items = data_wrangling.load_testset('train_r100')['item']
        mn_r100['item'] = items

        n = len(mn_r100['item'])
        np_ort = np.zeros(shape=(n, self.ort_units))
        np_pho = np.zeros(shape=(n, self.pho_units))
        np_sem = np.zeros(shape=(n, self.sem_units))

        for idx, item in enumerate(items):
            np_ort[idx,:] = np.array(self.ort_pattern[item])
            np_pho[idx,:] = np.array(self.pho_pattern[item])
            np_sem[idx,:] = self.sparse_to_dense(self.sparse_sem_pattern[item], self.sem_units)

        mn_r100['ort'] = np_ort
        mn_r100['pho'] = np_pho
        mn_r100['sem'] = np_sem
        mn_r100['cond'] = None

        data_wrangling.save_testset(mn_r100, 'dataset/testsets/mn_r100.pkl.gz')



    @staticmethod
    def get_pattern(lines, line_start, line_end):
        """Get unit activations from multiple lines"""
        pattern = []
        for i in range(line_start, line_end):
            line = lines[i].strip().split(' ')
            [pattern.append(int(unit)) for unit in line if unit.isdigit()]
        return pattern

    @staticmethod
    def sparse_to_dense(representation, units=2446) -> np.array:
        """Convert sparse representation to dense representation"""
        dense = np.zeros(units)
        for unit in representation:
            dense[int(unit)] = 1
        return dense

                

# %% Do the works

mn_pattern = MikeNetPattern()
mn_pattern.save()