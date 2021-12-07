from typing import List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import modeling
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """EnvironmentConfig Class constains all the experience related information"""

    # TODO: Add better support multiple stages environment and non-stationary support

    task_names: tuple = None
    wf_compression: str = None
    wf_clip_low: int = None
    wf_clip_high: int = None
    total_sample: int = None
    batch_size: int = None
    # tasks_ps_oral: int = None
    # tasks_ps_reading: int = None

    tasks_ps: tuple = None
    # tasks_start_ps: tuple = None
    # tasks_end_ps: tuple = None

    @classmethod
    def from_dict(cls, globals_dict: dict):
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
        A task object that contains the information of a single task.
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

        if progress_start is None:
            self.progress_start = 100  # Full open if not specified

        if progress_slope is None:
            self.progress_slope = 0  # Stationary if not specified

    def get_progress(self, sample: float) -> float:
        """Return the progress of a given sample in percentage"""
        progress = self.progress_start + sample * self.progress_slope
        return min(progress, 100)

    def __str__(self):
        return f"{self.name}: ({self.input_name} -> {self.output_name}), Corpus opening curve: {self.progress_slope} * sample + {self.progress_start}"


class Stage:
    """Stage contains multiple tasks and the probability of choosing a task."""

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
        self.task_probability_end = task_probability_end

        if task_probability_end is None:
            self.task_probability_end = task_probability_start

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
    def nonstationary_pretrain(cls, config: EnvironmentConfig):
        """Non-stationary pretraining environment constructor"""
        stages = []

        # Ramp-up stage
        speed = 100 / 10_000_000  # 100% per 10M sample
        tasks_s1 = [
            Task(x, progress_start=5, progress_slope=speed) for x in config.task_names
        ]

        stages.append(
            Stage(
                name="rampup",
                tasks=tasks_s1,
                stage_sample=4_500_000,
                task_probability_start=config.tasks_ps,
            )
        )

        # Sustain stage
        tasks_s2 = [Task(x, progress_start=50) for x in config.task_names]

        stages.append(
            Stage(
                name="sustain",
                tasks=tasks_s2,
                stage_sample=config.total_sample - 4_500_000,
                task_probability_start=config.tasks_ps,
            )
        )

        return cls(stages)

    @classmethod
    def stationary_from_config(cls, config: EnvironmentConfig):
        """Create a new experience from config"""

        tasks = [Task(x) for x in config.task_names]
        stages = [
            Stage("one", tasks, config.total_sample, config.tasks_ps)
        ]  # Stationary

        return cls(stages)

    @classmethod
    def non_stationary_from_config(cls, config: EnvironmentConfig):
        """Create a new experience from config"""

        stages = []
        # Transition stage (1M)
        ## Task mixing changes linear from start (oral) to end (reading)
        ## And keep the corpus ramp up at a constant speed

        transition_sample = 1_000_000
        speed = 100 / 5_000_000  # 100% per 5M sample
        tasks_s1 = [
            Task(x, progress_start=50, progress_slope=speed)
            if x != "triangle"
            else Task(x, progress_start=0, progress_slope=speed)
            for x in config.task_names
        ]

        stages.append(
            Stage(
                name="transition",
                tasks=tasks_s1,
                stage_sample=transition_sample,
                task_probability_start=config.tasks_ps_oral,
                task_probability_end=config.tasks_ps_reading,
            )
        )

        # Reading stage (Remaining)
        s2_oral_start_pc = 50 + speed * (transition_sample)
        s2_reading_start_pc = 0 + speed * (transition_sample)

        tasks_s2 = [
            Task(x, progress_start=s2_oral_start_pc, progress_slope=speed)
            if x != "triangle"
            else Task(x, progress_start=s2_reading_start_pc, progress_slope=speed)
            for x in config.task_names
        ]

        stages.append(
            Stage(
                name="reading",
                tasks=tasks_s2,
                stage_sample=config.total_sample - transition_sample,
                task_probability_start=config.tasks_ps_reading,
            )
        )

        return cls(stages)

    def plot_corpus(self, scale_x=10000):
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

    def plot_task_probability(self, scale_x=10000):
        """Plot task probability in each stage"""
        fig = plt.figure()

        for i, stage in enumerate(self.stages):
            samples = stage.stage_sample
            ps = [
                stage.get_task_probability(sample * scale_x)
                for sample in range(int(samples / scale_x))
            ]
            ax = fig.add_subplot(1, self.n_stages, i + 1)

            for j, task in enumerate(stage.tasks):
                task_p = [p[j] for p in ps]
                ax.plot(task_p, label=task.name)
                ax.legend(loc="lower right")
                ax.set_title(f"Stage {i+1}: Sampling probability in {stage.name}")
                ax.set_xlabel(f"sample x {scale_x}")
                ax.set_ylabel("Probability")
                ax.set_ylim([0, 1])

    def get_stage(self, sample: int) -> Tuple[Stage, int]:
        """Get the current stage and sample_at_stage by no. sample ingested
        Stage: Stage object
        sample_at_stage: the no. of sample counted from the start of a stage
        """
        cumulative_sample = 0
        sample_at_stage = sample

        # Walk through all stages and return the stage and sample_at_stage
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
            words = self.data.df_train.word[idx]

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
            yield task.name, idx, words, batch_x, batch_y
