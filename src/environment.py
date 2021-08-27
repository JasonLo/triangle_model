from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
import modeling
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """EnvironmentConfig Class constains all the experience related information"""

    # TODO: Add support multiple stages and non-stationary support

    task_names: tuple = None
    wf_compression: str = None
    wf_clip_low: int = None
    wf_clip_high: int = None
    total_sample: int = None
    tasks_ps: tuple = None
    batch_size: int = None

    @classmethod
    def from_global(cls, globals_dict):
        config_dict = {
            k: globals_dict[k] for k in globals_dict if k in cls.__annotations__.keys()
        }
        return cls(**config_dict)


class Task:
    """A single task that supports nonstaionary environment"""

    def __init__(
        self,
        name: str,
        input_name: str = None,
        output_name: Union[str, List[str]] = None,
        progress_start: float = None,
        progress_slope: float = None,
    ):
        """
        progress_start: starting location
        progress_slope: slope progress in %/sample
        """
        self.name = name
        self.input_name = (
            input_name if input_name is not None else modeling.IN_OUT[name][0]
        )
        self.output_name = (
            output_name if output_name is not None else modeling.IN_OUT[name][1]
        )
        self.progress_start = (
            progress_start if progress_start is not None else 100
        )  # Full open if not specified
        self.progress_slope = (
            progress_slope if progress_slope is not None else 0
        )  # Stationary if not specified

    def get_progress(self, sample: float) -> float:
        """Return the progress of a given sample in percentage"""
        progress = self.progress_start + sample * self.progress_slope
        return min(progress, 100)

    def __str__(self):
        return f"{self.name}: ({self.input_name} -> {self.output_name}), Corpus opening curve: {self.progress_slope} * sample + {self.progress_start}"


class Stage:
    """Stage contains multiple tasks and the probability of choosing a task"""

    def __init__(
        self,
        name: str,
        tasks: List[Task],
        stage_sample: int,
        task_probability_start: List[float],
        task_probability_end: List[float] = None,
    ):
        self.name = name
        self.tasks = tasks
        self.stage_sample = stage_sample
        self.task_probability_start = task_probability_start
        self.task_probability_end = (
            task_probability_start
            if task_probability_end is None
            else task_probability_end
        )

    def get_task_probability(self, sample: int) -> float:
        """Return the probability of a task at a given sample"""
        interpolate = lambda sample, start, end: np.interp(
            sample, xp=[0, self.stage_sample], fp=[start, end]
        )
        p = [
            interpolate(sample, start, end)
            for start, end in zip(
                self.task_probability_start, self.task_probability_end
            )
        ]
        return p

    def draw_task(self, sample: int) -> Task:
        """Return the task to be executed"""
        p = self.get_task_probability(sample)
        task = np.random.choice(self.tasks, p=p)
        return task


class Experience:
    """Experience contains all the stages a model will go through"""

    def __init__(self, stages: List[Stage]):
        self.stages = stages
        self.n_stages = len(stages)
        self.total_sample = sum(x.stage_sample for x in self.stages)

    @classmethod
    def from_config(cls, config: EnvironmentConfig):
        """Create a new experience from config"""

        tasks = [Task(x) for x in config.task_names]
        stages = [
            Stage("one", tasks, config.total_sample, config.tasks_ps)
        ]  # Stationary

        return cls(stages)

    def plot_corpus(self, scale_x=1000):
        """Plot the corpus opening progress in each stage"""
        fig = plt.figure()

        for i, stage in enumerate(self.stages):
            samples = stage.stage_sample
            ax = fig.add_subplot(1, self.n_stages, i + 1)
            for j, task in enumerate(stage.tasks):
                y = [
                    task.get_progress(sample * scale_x)
                    for sample in range(int(samples / scale_x))
                ]
                ax.plot(y, label=task.name)
                ax.legend(loc="lower right")
                ax.set_title(
                    f"Stage {i+1}: Corpus opening progression (%) in {stage.name}"
                )
                ax.set_xlabel(f"sample x {scale_x}")
                ax.set_ylabel("Progress (%)")
                ax.set_ylim([0, 101])

    def plot_task_probability(self, scale_x=1000):
        """Plot task probability in each stage"""
        fig = plt.figure()

        for i, stage in enumerate(self.stages):
            samples = stage.stage_sample
            ps = [
                stage.get_task_probability(sample * scale_x)
                for sample in range(int(samples / scale_x))
            ]
            ax = fig.add_subplot(self.n_stages, 1, i + 1)

            for j, task in enumerate(stage.tasks):
                task_p = [p[j] for p in ps]
                ax.plot(task_p, label=task.name)
                ax.legend(loc="lower right")
                ax.set_title(f"Stage {i+1}: Sampling probability in {stage.name}")
                ax.set_xlabel(f"sample x {scale_x}")
                ax.set_ylabel("Probability")
                ax.set_ylim([0, 1])

    def get_stage(self, sample: int) -> [Stage, int]:
        """Get the current stage and sample_at_stage by no. sample ingested
        Stage: Stage object
        sample_at_stage: the no. of sample counted from the start of a stage
        """
        cumulative_sample = 0
        sample_at_stage = sample

        for stage in self.stages:
            if sample <= cumulative_sample + stage.stage_sample:
                return stage, sample_at_stage
            else:
                cumulative_sample += stage.stage_sample
                sample_at_stage = sample - cumulative_sample


class Sampler:
    """v3 Sampler for modularized environment
    Features: Fully modularized environment staging
    """

    def __init__(self, cfg, data, experience):

        # Get necessary environment config from cfg object
        self.cfg = cfg

        # Expose data object to higher level
        self.wf_clip_low = cfg.wf_clip_low
        self.wf_clip_high = cfg.wf_clip_high
        self.wf_compression = cfg.wf_compression
        self.batch_size = cfg.batch_size
        self.n_timesteps = cfg.n_timesteps
        self.inject_error_ticks = cfg.inject_error_ticks

        np.random.seed(cfg.rng_seed)

        self.data = data
        self.current_sample = 0
        self.experience = experience

        # Basic convenient variables
        self._calculate_aux_variables()

    def _calculate_aux_variables(self):

        # Word frequency related
        if self.wf_clip_low is None:
            self.wf_clip_low = 0

        if self.wf_clip_high is None:
            self.wf_clip_high = 999999999

        clip_wf = self.data.df_train.wf.clip(self.wf_clip_low, self.wf_clip_high).copy()
        self.rank_pct_wf = clip_wf.rank(pct=True, ascending=False)
        self.rank_pct_wf_dict = dict(zip(self.data.df_train.word, self.rank_pct_wf))

        assert self.wf_compression in ("log", "root")
        self.compressed_wf = (
            np.log(clip_wf + 1) if self.wf_compression == "log" else np.sqrt(clip_wf)
        )

    def wf_to_ps(self, wf):
        """convert squashed compressed word frequency to probability"""
        return np.array(wf / np.sum(wf), dtype="float32")

    def sample_to_batch(self, sample):
        """Convert sample to batch in 0-indexing format"""
        return int(sample / self.batch_size)

    def get_sampling_p(self, task, stage_sample):
        """Get sampling probability for a task"""
        progress = task.get_progress(stage_sample)
        # Create selection mask
        mask = self.rank_pct_wf < progress
        return self.wf_to_ps(self.compressed_wf * mask)

    def generator(self):
        """Generator that draw task and sample idx"""
        x_ticks = self.n_timesteps
        y_ticks = self.inject_error_ticks

        while True:
            stage, stage_sample = self.experience.get_stage(self.current_sample)
            task = stage.draw_task(stage_sample)
            idx = np.random.choice(
                self.data.df_train.index,
                self.batch_size,
                p=self.get_sampling_p(task, stage_sample),
            )

            x, y = modeling.IN_OUT[task.name]
            batch_x = [self.data.np_representations[x][idx]] * x_ticks

            # Check if multiple output or not
            if type(y) is list:
                batch_y = {
                    yi: [self.data.np_representations[yi][idx]] * y_ticks for yi in y
                }
            else:
                batch_y = [self.data.np_representations[y][idx]] * y_ticks

            self.current_sample += self.batch_size
            yield task.name, idx, batch_x, batch_y