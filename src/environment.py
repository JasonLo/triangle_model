from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from modeling import IN_OUT


@dataclass
class EnvironmentConfig:
    """EnvironmentConfig Class constains all the experience related information.
    IMPORTANT: Only contain model level environment config information (No staging details).
    Becuase staging details are not serializable (yet), they are not included in this class.

    TODO: Make this EnvironmentConfig() and Experience() interopable, where Experience()
    can be constructed from EnvironmentConfig() and vice versa.

    Arguements:
        task_names
    """

    wf_compression: str = None
    wf_clip_low: int = None
    wf_clip_high: int = None
    task_names: tuple = None
    tasks_ps: tuple = None
    total_sample: int = None

    @classmethod
    def from_dict(cls, d: dict):
        config_dict = {k: d[k] for k in d if k in cls.__annotations__.keys()}
        return cls(**config_dict)


class Task:
    """A single task that supports nonstaionary environment."""

    def __init__(
        self, name: str, progress_start: float = None, progress_slope: float = None
    ):
        """
        A task object that contains the information of a single task.
        Arguments:
            name: Name of the task.
            progress_start: Starting point of the corpus opening progress curve.
            progress_slope: Slope of the corpus opening progress curve (in % per sample).
        """
        self.name = name
        self.progress_start = progress_start if progress_start is not None else 100
        self.progress_slope = progress_slope if progress_slope is not None else 0

    def get_progress(self, sample: float) -> float:
        """Return the progress of a given sample in percentage."""
        progress = self.progress_start + sample * self.progress_slope
        return min(progress, 100)

    def __repr__(self):
        return f"{self.name}: Corpus opening curve: {self.progress_slope} * sample + {self.progress_start}"


class Stage:
    """Stage contains multiple tasks and the probability of choosing a task.

    Arguments:
        name: Name of the stage.
        tasks: List of tasks in the stage.
        stage_sample: Number of samples in the stage.
        task_probability_start: List of probabilities of choosing a task at the beginning of the stage.
        task_probability_end (optional): List of probabilities of choosing a task at the end of the stage.
            If not provided, task_probability will remain constant (= task_probability_start) within the stage.
    """

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

    def get_task_probability(self, sample: int) -> list:
        """Return the probability of a task at a given sample."""

        interpolate = lambda sample, start, end: np.interp(
            sample, xp=[0, self.stage_sample], fp=[start, end]
        )

        return [
            interpolate(sample, start, end)
            for start, end in zip(
                self.task_probability_start, self.task_probability_end
            )
        ]

    def draw_task(self, sample: int) -> Task:
        """Return the task to be use."""
        p = self.get_task_probability(sample)
        task = np.random.choice(self.tasks, p=p)
        return task


class Experience:
    """Experience contains all the stages a model will go through.

    Arguments:
        stages: a list of Stage objects

    Examples:

        #################### Create a HS04-like experience #####################
        ```
        oral_tasks = [Task(x) for x in ["pho_sem", "sem_pho", "pho_pho", "sem_sem"]]
        oral_stage = Stage(
            name="hs04-oral",
            tasks=oral_tasks,
            stage_sample=700_000,
            task_probability_start=[0.4, 0.4, 0.1, 0.1],
        )

        reading_tasks = [Task("triangle")]
        reading_stage = Stage(
            name="hs04-reading",
            tasks=reading_tasks,
            stage_sample=1_500_000,
            task_probability_start=[1.0],
        )

        hs04_experience = Experience(stages=[oral_stage, reading_stage])
        ```

        ########### Create a non-stationary single stage experience ############
        - The task probability is transitioning to all oral to 50% oral and 50% reading.
        - Oral tasks corpus have a 50% opening at start, then increase by a rate of 100%/M samples.
        - Reading tasks corpus have a 0% opening at start, then increase by a rate of 100%/M samples.
        ```
        speed = 100 / 1_000_000
        oral_task_names = ["pho_sem", "sem_pho", "pho_pho", "sem_sem"]
        oral_tasks = [Task(x, progress_start = 50, progress_slope = speed) for x in oral_task_names]
        triangle_task = [Task("triangle", progress_start = 0, progress_slope = speed)]

        one_stage = Stage(
            name = "non-stationary",
            tasks = oral_tasks + triangle_task,
            stage_sample = 1_000_000,
            task_probability_start = [0.4, 0.4, 0.1, 0.1, 0.0],
            task_probability_end = [0.2, 0.2, 0.05, 0.05, 0.5],
        )

        ns_experience = Experience(stages=[one_stage])
        ```
    """

    def __init__(self, stages: List[Stage]):
        self.stages = stages
        self.n_stages = len(stages)
        self.total_sample = sum(x.stage_sample for x in self.stages)

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
                ax.set_ylim([-3, 103])  # Easier to see boundary case

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
                ax.set_ylim([-0.03, 1.03])  # Easier to see boundary case

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
    Features: Fully support Experience().
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

        clip_wf = self.data["wf"].clip(self.wf_clip_low, self.wf_clip_high).copy()
        self.rank_pct_wf = clip_wf.rank(pct=True, ascending=False)
        self.rank_pct_wf_dict = dict(zip(self.data["item"], self.rank_pct_wf))

        assert self.wf_compression in ("log", "root")
        self.compressed_wf = (
            np.log(clip_wf + 1) if self.wf_compression == "log" else np.sqrt(clip_wf)
        )

    def wf_to_ps(self, wf):
        """convert squashed compressed word frequency to probability."""
        return np.array(wf / np.sum(wf), dtype="float32")

    def sample_to_batch(self, sample):
        """Convert sample to batch in 0-indexing format."""
        return int(sample / self.batch_size)

    def get_sampling_p(self, task, stage_sample):
        """Get sampling probability for a task."""
        progress = task.get_progress(stage_sample)
        # Create selection mask
        mask = self.rank_pct_wf < progress
        return self.wf_to_ps(self.compressed_wf * mask)

    def generator(self):
        """Generator that draw task and sample idx."""
        x_ticks = self.n_timesteps
        y_ticks = self.inject_error_ticks

        while True:
            stage, stage_sample = self.experience.get_stage(self.current_sample)
            task = stage.draw_task(stage_sample)
            idx = np.random.choice(
                self.data["id"],
                self.batch_size,
                p=self.get_sampling_p(task, stage_sample),
            )
            words = self.data["item"][idx]

            x, y = IN_OUT[task.name]
            batch_x = [self.data[x][idx]] * x_ticks

            # Check if multiple output or not
            if type(y) is list:
                batch_y = {yi: [self.data[yi][idx]] * y_ticks for yi in y}
            else:
                batch_y = [self.data[y][idx]] * y_ticks

            self.current_sample += self.batch_size
            yield task.name, idx, words, batch_x, batch_y
