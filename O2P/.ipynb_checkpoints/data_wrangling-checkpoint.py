
import pandas as pd
import numpy as np

def gen_pkey(p_file="patterns/mapping_v2.txt"):
    
    # read phonological patterns from the mapping file
    # See Harm & Seidenberg PDF file
    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict('list')
    return m_dict

def data_wrangling():

    # Read training file
    train_file = 'patterns/6ktraining.dict'

    strain_file = 'patterns/strain.txt'  
    strain_key_file='patterns/strain_key.txt'

    grain_file = 'patterns/grain_nws.dict'
    grain_key_file='patterns/grain_key.txt'


    train = pd.read_csv(train_file,sep='\t', header=None, names=['word', 'ort', 'pho', 'wf'])

    strain = pd.read_csv(strain_file,sep='\t', header=None, names=['word', 'ort', 'pho', 'wf'])
    strain_key = pd.read_table(strain_key_file, header=None, delim_whitespace=True, names=['word', 'frequency', 'pho_consistency', 'imageability'])
    df_strain = pd.merge(strain, strain_key)

    grain = pd.read_csv(grain_file,sep='\t', header=None, names=['word', 'ort', 'pho_large', 'pho_small'])
    grain_key = pd.read_table(grain_key_file, header=None, delim_whitespace=True, names=['word', 'condition'])
    grain_key['condition'] = np.where(grain_key['condition']=='critical', 'ambiguous', 'unambiguous')
    df_grain = pd.merge(grain, grain_key)

    def prepDF(t):
        # The first bit and last 3 bits are empty in this dataset
        t['ort'] = t.ort.apply(lambda x: x[1:11])
        return t

    df_train = prepDF(train)
    df_strain = prepDF(df_strain)
    df_grain = prepDF(df_grain)

    df_train.to_csv('input/df_train.csv')
    df_strain.to_csv('input/df_strain.csv')
    df_grain.to_csv('input/df_grain.csv')

    print(df_train.head)
    print(df_strain.head)
    print(df_grain.head)

    # Encode orthographic representation
    def ort2bin(o_col, trimMode=True, verbose=True):
        # Replicating support.py (o_char)
        # This function wrap tokenizer.texts_to_matrix to fit on multiple 
        # independent slot-based input
        # i.e. one-hot encoding per each slot with independent dictionary

        from tensorflow.keras.preprocessing.text import Tokenizer

        nSlot = len(o_col[0])
        nWord = len(o_col)

        slotData = nWord*[None]
        binData = pd.DataFrame()

        for slotId in range(nSlot):
            for wordId in range(nWord):
                slotData[wordId] = o_col[wordId][slotId]

            t = Tokenizer(filters='', lower=False)
            t.fit_on_texts(slotData)
            seqData = t.texts_to_sequences(slotData)  # Maybe just use sequence data later
            
            # Triming first bit in each slot
            if trimMode == True:
                tmp = t.texts_to_matrix(slotData)
                thisSlotBinData = tmp[:,1::]   # Remove the first bit which indicate a separate slot (probably useful in recurrent network)
            elif trimMode == False:
                thisSlotBinData = t.texts_to_matrix(slotData)

            # Print dictionary details
            if verbose == True:
                print('In slot ', slotId, '\t')
                print('token count:', t.word_counts)
                print('word count:', t.document_count)
                print('dictionary:', t.word_index)
                print('token appear in how many words:', t.word_docs)

            # Put binary data into a dataframe
            binData = pd.concat([binData, pd.DataFrame(thisSlotBinData)], axis=1, ignore_index=True)

        return binData

    def ort2bin_v2(o_col):
        # Use tokenizer instead to acheive same thing, but with extra zeros columns
        from tensorflow.keras.preprocessing.text import Tokenizer
        t = Tokenizer(filters='', lower=False, char_level=True)
        t.fit_on_texts(o_col)
        print('dictionary:', t.word_index)
        return t.texts_to_matrix(o_col)

    # Merge all 3 ortho representation
    all_ort = pd.concat([df_train.ort, df_strain.ort, df_grain.ort], ignore_index=True)

    # Encoding orthographic representation
    all_ort_bin     = ort2bin(all_ort, verbose=False)
    splitId_strain  = len(df_train)
    splitId_grain   = len(df_train) + len(df_strain)

    x_train     = np.array(all_ort_bin[0:splitId_strain])
    x_strain    = np.array(all_ort_bin[splitId_strain:splitId_grain])
    x_grain     = np.array(all_ort_bin[splitId_grain::])

    # Save to disk
    np.savez_compressed('input/x_train.npz', data = x_train)
    np.savez_compressed('input/x_strain.npz', data = x_strain)
    np.savez_compressed('input/x_grain.npz', data = x_grain)

    print('==========Orthographic representation==========')
    print('all shape:', all_ort_bin.shape)
    print('x_train shape:', x_train.shape)
    print('x_strain shape:', x_strain.shape)
    print('x_grain shape:', x_grain.shape)

    # Encode phonological representation

    def pho2bin(p_col, p_key):
        # Was called p_pattern in original support script
        phoneme_len = len(p_key["p"])  # just chose an item at random for the length as they are all equal
        p_output = np.zeros((len(p_col), len(p_col[0])*phoneme_len))  # same true for item 0 here

        # iterate through items
        for i in range(0, len(p_col)):
            word_now = list(p_col[i])  # separate input into characters
            whole_word = []

            # loop through characters
            for j in word_now:  # convert char to mapping
                slot = p_key[j]
                whole_word.append(slot)  # append to the item

            p_output[i] = np.concatenate(whole_word)
        return p_output

    def pho2bin_v2(p_col, p_key):
        # Vectorize for performance (that no one ask for... )
        binLength = len(p_key['_'])
        n = len(p_col)
        nPhoChar = len(p_col[0])

        p_output = np.empty([n, binLength*nPhoChar])

        for slot in range(len(p_col[0])):
            slotSeries = p_col.str.slice(start=slot, stop=slot+1)
            outSeries = slotSeries.map(p_key)
            p_output[:,range(slot*25, (slot+1)*25)] = outSeries.to_list()
        return p_output
    
    phon_key = gen_pkey()
    y_train = pho2bin_v2(train.pho, phon_key)
    y_strain = pho2bin_v2(strain.pho, phon_key)
    y_large_grain = pho2bin_v2(grain.pho_large, phon_key)
    y_small_grain = pho2bin_v2(grain.pho_small, phon_key)

    # Save to disk
    np.savez_compressed('input/y_train.npz', data = y_train)
    np.savez_compressed('input/y_strain.npz', data = y_strain)
    np.savez_compressed('input/y_large_grain.npz', data = y_large_grain)
    np.savez_compressed('input/y_small_grain.npz', data = y_small_grain)

    print('\n==========Phonological representation==========')
    print(len(phon_key), ' phonemes: ', phon_key.keys())
    print('y_train shape:', y_train.shape)
    print('y_strain shape:', y_strain.shape)
    print('y_large_grain shape:', y_large_grain.shape)
    print('y_small_grain shape:', y_small_grain.shape)

# data_wrangling()
class wfManager():
    import numpy as np
    # Note: the probability must sum to 1 when passing it to np.random.choice()
    def __init__(self, wf):
        self.wf = np.array(wf)

    def wf(self): return self.wf

    def to_p(self, x): return x/np.sum(x)

    def samp_termf(self):
        return self.to_p(self.wf)

    def samp_log(self):
        log = np.log(1+self.wf)
        return self.to_p(log)

    def samp_hs04(self):
        root = np.sqrt(self.wf)/np.sqrt(30000)
        clip = root.clip(0.05,1.0)
        return self.to_p(clip)

    def samp_jay(self):
        cap = self.wf.clip(0, 10000)
        root = np.sqrt(cap)
        return self.to_p(root)

def sampleGenerator(x_set, y_set, n_timesteps, batch_size, sample_p, rng_seed):
# Get <batch_size> of data from <x_set>, <y_set> based on the probability of <sample_p>
    np.random.seed(rng_seed)
    while 1:
        idx = np.random.choice(range(len(sample_p)), batch_size, p=sample_p)
        batch_x = x_set[idx]
        batch_y = []
        
        for i in range(n_timesteps):
            batch_y.append(y_set[idx])
        yield (batch_x, batch_y)

def sample_generator_with_semantic(cfg, data):
    # Get <batch_size> of data from <x_set>, <y_set> based on the probability of <sample_p>
    from data_wrangling import semantic_influence
    import tensorflow as tf 
    
    np.random.seed(cfg.sample_rng_seed)
    t = 0
    batch = 0
    
    while True:
        batch += 1
        idx = np.random.choice(range(len(data.sample_p)), cfg.batch_size, p=data.sample_p)
        batch_x = data.x_train[idx]
        
        batch_y = []
        for i in range(cfg.n_timesteps):
            batch_y.append(data.y_train[idx])
            
        input_s_cell = semantic_influence(t, data.wf[idx])
        input_s = tf.tile(tf.expand_dims(input_s_cell, 1), [1, cfg.pho_units])
        if batch % cfg.steps_per_epoch == 0: t += 1  # Recording time for semantic ramp up
        # print('Time = {}'.format(t)) 
        
        yield ([batch_x, input_s], batch_y)
        
        
class my_data():
    def __init__(self, cfg):
        import numpy as np
        import pandas as pd

        self.sample_name = cfg.sample_name

        self.df_train = pd.read_csv('input/df_train.csv', index_col=0)
        self.df_strain = pd.read_csv('input/df_strain.csv', index_col=0)
        self.df_grain = pd.read_csv('input/df_grain.csv', index_col=0)

        self.x_train = np.load('input/x_train.npz')['data']
        self.x_strain = np.load('input/x_strain.npz')['data']
        self.x_grain = np.load('input/x_grain.npz')['data']
        self.x_grain_wf = np.array(self.df_strain['wf'])
        
        self.y_train = np.load('input/y_train.npz')['data']
        self.y_strain = np.load('input/y_strain.npz')['data']
        self.y_large_grain = np.load('input/y_large_grain.npz')['data']
        self.y_small_grain = np.load('input/y_small_grain.npz')['data']

        from data_wrangling import gen_pkey
        self.phon_key = gen_pkey('input/mapping_v2.txt')

        print('==========Orthographic representation==========')
        print('x_train shape:', self.x_train.shape)
        print('x_strain shape:', self.x_strain.shape)
        print('x_grain shape:', self.x_grain.shape)

        print('\n==========Phonological representation==========')
        print(len(self.phon_key), ' phonemes: ', self.phon_key.keys())
        print('y_train shape:', self.y_train.shape)
        print('y_strain shape:', self.y_strain.shape)
        print('y_large_grain shape:', self.y_large_grain.shape)
        print('y_small_grain shape:', self.y_small_grain.shape)

        self.gen_sample_p()
        self.wf = np.array(self.df_train['wf'], dtype='float32')

    def gen_sample_p(self):
        from data_wrangling import wfManager
        wf = wfManager(self.df_train['wf'])

        if self.sample_name == 'hs04': self.sample_p = wf.samp_hs04()
        if self.sample_name == 'jay': self.sample_p = wf.samp_jay()
        if self.sample_name == 'log': self.sample_p = wf.samp_log()

            
def bq_conn(pid='idyllic-web-267716'):
    from google.oauth2 import service_account
    project_id = pid
    credentials = service_account.Credentials.from_service_account_file(
                 '../bq_credential.json')
    
def write_cfg_to_bq(cfg):
    import pandas as pd
    import pandas_gbq

    pandas_gbq.to_gbq(pd.DataFrame([cfg.cfg_dict]), 
                      destination_table='batch_test.cfg2', 
                      project_id='idyllic-web-267716', 
                      if_exists='append')
        
def semantic_influence(t, f, g=5., k=2000.):
    import tensorflow as tf
    lt = tf.math.multiply(tf.math.log(tf.math.add(f,2.)),t)
    return tf.math.divide(tf.math.multiply(g,lt),tf.math.add(lt,k))
