# Develop an integrated sampler for TF model 4.0
"""
Requirements:
1. Smooth transition of corpus opening
2. Smooth transition of task proportion changing

Staging:
I: Oral
II: Transition (start moving from oral_tasks_ps to reading_tasks_ps)
III: Reading (Full usage of reading_tasks_ps)

Parameters:
tasks: Master task list in all stage, probability follows this order (e.g., ("pho_sem", "sem_pho", "pho_pho", "sem_sem", "triangle"))
wf_clip_low: Low word frequency clipping
wf_clip_high: High word frequency clipping
wf_compression: Word frequency compression ("log" or "root") 
oral_plateau: How many sample will oral stage fully open its corpus
oral_tasks_ps: Probability of task selection during oral stage (e.g., (0.4, 0.4, 0.1, 0.1, 0.0))
reading_tasks_ps: Probability of task selection during reading stage (e.g., (0.2, 0.2, 0.05, 0.05, 0.5))
stage_lag: How much reading task corpus opening lag behind oral, measured by percentage
transition_samples: How many sample in transition stage
"""

# %%
import numpy as np
import data_wrangling, modeling

data = data_wrangling.MyData()

env_cfg = {
    "tasks": ("pho_sem", "sem_pho", "pho_pho", "sem_sem", "triangle"),
    "wf_clipping_edges": None,
    "wf_compression": 'log',
    "oral_start_pct": 0.02,
    "oral_end_pct": .5, 
    "oral_sample": 900_000,
    "oral_tasks_ps": (0.4, 0.4, 0.1, 0.1, 0.0),
    "transition_sample": 400_000,
    "reading_sample": 2_000_000,
    "reading_tasks_ps": (0.2, 0.2, 0.05, 0.05, 0.5),
    "batch_size": 10000,
    "rng_seed": 2021
}

# %%
class Sampler:

    def __init__(self, env_cfg, data):

        # environment config (Init for style)
        self.tasks = None
        self.wf_clip_low = None
        self.wf_clip_high = None
        self.wf_compression = None
        self.oral_start_pct = None
        self.oral_end_pct = None 
        self.oral_sample = None
        self.oral_tasks_ps = None
        self.transition_sample = None
        self.reading_sample = None
        self.reading_tasks_ps = None
        self.batch_size = None

        # Ingest all values from dictionary
        for key, value in env_cfg.items():
            setattr(self, key, value)

        self.data = data
        self.current_sample = 0

        # Basic convienient variables
        self._calculate_aux_variables()

        # Task probability dictionary (batch, ps)
        self.ps = {}
        self._calculate_task_ps()

        # Progress dictionary 
        self.progress = {}
        self._calculate_progress_dict()

    def _calculate_aux_variables(self):
        self.total_sample = self.oral_sample + self.reading_sample

        self.total_batches = self.sample_to_batch(self.total_sample)
        self.oral_batches = self.sample_to_batch(self.oral_sample)
        self.transition_batches = self.sample_to_batch(self.transition_sample)
        self.remaining_batches = self.total_batches - self.oral_batches - self.transition_batches

        # Word frequency related
        clip_wf = self.data.df_train.wf.clip(self.wf_clip_low, self.wf_clip_high).copy()
        self.rank_pct_wf = clip_wf.rank(pct=True, ascending=False)

        assert self.wf_compression in ('log', 'root')
        self.compressed_wf = np.log(clip_wf + 1) if self.wf_compression == 'log' else np.sqrt(clip_wf)

    def _calculate_task_ps(self):
        """Task probabililty store all task probabilities in a single dictionary (epoch, ps) """

        # Oral
        self.task_ps = {i:self.oral_tasks_ps for i in range(self.oral_batches)}

        # Transition
        tps = np.linspace(self.oral_tasks_ps, self.reading_tasks_ps, self.transition_batches)
        tran_ps = {i + self.oral_batches: tuple(tps[i]) for i in range(self.transition_batches)}
        self.task_ps.update(tran_ps)

        # Remaining
        remain_ps = {i + self.oral_batches + self.transition_batches: self.reading_tasks_ps for i in range(self.remaining_batches)}
        self.task_ps.update(remain_ps)

    def _calculate_progress_dict(self):
        """Progress dictionary storing all %max progress at each epoch in oral and reading stage"""
        # Oral progress
        opg = np.linspace(self.oral_start_pct, self.oral_end_pct, self.oral_batches)
        self.progress['oral'] = self._extrapolate_ps(opg)

        # Reading progress
        self.progress['reading'] = self._right_shift_ps(self.progress['oral'])         

    def _extrapolate_ps(self, array):
        """Extrapolate an array to the number of batches long"""
        d = array[1] - array[0]
        e = array[-1]
        n = len(array)
        ex_array = [array[i] if i < n  else e + (i-n) * d for i in range(self.total_batches)]
        return np.clip(ex_array, 0, 1)

    def _right_shift_ps(self, oral_progress):
        """Convert oral ps to reading ps"""
        n = self.oral_batches
        beginning = np.zeros(n)
        return np.concatenate([beginning, oral_progress[:-n]])

    def wf_to_ps(self, wf):
        """convert squashed compressed word frequncy to probabilty"""
        return np.array(wf/np.sum(wf), dtype="float32")

    def sample_to_batch(self, sample):
        """Convert sample to batch in 0-indexing format"""
        return int(sample/self.batch_size)

    def get_sampling_p(self, current_sample, task):
        current_batch = self.sample_to_batch(current_sample)

        if task == 'triangle':
            progress = self.progress['reading'][current_batch]
        else:
            progress = self.progress['oral'][current_batch]

        # Create selection mask
        mask = self.rank_pct_wf < progress
    
        return self.wf_to_ps(self.compressed_wf * mask)

    def generator(self, x_ticks, y_ticks):
        """ Generator that draw task and sample idx 
        """
        
        while True:
            current_batch = self.sample_to_batch(self.current_sample)
            task = np.random.choice(self.tasks, p=self.task_ps[current_batch])
            idx = np.random.choice(self.data.df_train.index, self.batch_size, p=self.get_sampling_p(self.current_sample, task))
            
            x, y = modeling.IN_OUT[task]
            batch_x = [self.data.np_representations[x][idx]] * x_ticks

            if type(y) is list:
                    batch_y = [[self.data.np_representations[yi][idx]] * y_ticks for yi in y]
            else:
                # Single output
                batch_y = [self.data.np_representations[y][idx]] * y_ticks
            
            
            self.current_sample += self.batch_size
            yield task, idx, batch_x, batch_y
     


# %%
s = Sampler(env_cfg, data)
g = s.generator(3, 2)

task_list = []
idx_list = []

for i in range(s.total_batches):
    task, idx, bx, by = next(g)
    task_list.append(task)
    idx_list.append(idx)



# %%
task_list
# %%

bx[0].shape
# %%
by[0].shape
# %%
