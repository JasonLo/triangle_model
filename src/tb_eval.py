# Kind of obsolete in training loop testset
# hard to integrate and customize... maybe use offline eval again.
# 
# Usage: 
# 1. Create test object 
# strain = BaseTestSet(
#     cfg, 
#     name='strain', 
#     conds=('hf_con_hi', 'hf_con_li', 'hf_inc_hi', 'hf_inc_li', 'lf_con_hi', 'lf_con_li', 'lf_inc_hi', 'lf_inc_li'),
#     tasks=cfg.tasks
# )
# 
# 2. Add object call to end of epoch op: 
# [strain(task, step=epoch) for task in strain.tasks]
# 
# 3. Examine condition (run) x metrics (plot) value over epoch in tensorboard 

class BaseTestSet:
    """ Create a test set for tensorboard
    Dictionary nested structure: task > output (triangle only) > cond > metric
    """
    METRICS_MAP = {
            'pho_pho':{'acc': metrics.PhoAccuracy, 'sse': metrics.SumSquaredError},
            'sem_pho':{'acc': metrics.PhoAccuracy, 'sse': metrics.SumSquaredError},
            'pho_sem':{'acc': metrics.RightSideAccuracy, 'sse': metrics.SumSquaredError},
            'sem_sem':{'acc': metrics.RightSideAccuracy, 'sse': metrics.SumSquaredError},
            'triangle':{
                'pho': {'acc': metrics.PhoAccuracy, 'sse': metrics.SumSquaredError},
                'sem': {'acc': metrics.RightSideAccuracy, 'sse': metrics.SumSquaredError}
            }
        }

    
    def __init__(self, cfg, name, conds, tasks):
        self.cfg = cfg
        self.name = name
        self.conds = conds
        self.tasks = tasks
        self.testsets = {cond: data_wrangling.load_testset(os.path.join(self.cfg.tf_root, "dataset", "testsets", f"{self.name}_{cond}.pkl.gz")) for cond in self.conds}
        self.metrics = self._get_metrics()
        self.teststeps = {task: self.get_test_step(task) for task in self.tasks}

        # Use writer to differentiate condition
        self.writers = {cond: tf.summary.create_file_writer(os.path.join(cfg.path["tensorboard_folder"], 'test', self.name, cond)) for cond in self.conds}

    @staticmethod
    def get_test_step(task):
        if task == 'triangle':
            @tf.function()
            def test_step(x, y, model, metrics):
                # Unpacking for easier access
                y_pred_pho, y_pred_sem = model(x)
                y_pho, y_sem = y
                metrics_pho, metrics_sem = metrics

                [m.update_state(tf.cast(y_pho[-1], tf.float32), y_pred_pho[-1]) for m in metrics_pho]
                [m.update_state(tf.cast(y_sem[-1], tf.float32), y_pred_sem[-1]) for m in metrics_sem]

        else:
            @tf.function()
            def test_step(x, y, model, metrics):
                y_pred = model(x)
                [m.update_state(tf.cast(y[-1], tf.float32), y_pred[-1]) for m in metrics]
        return test_step

    def _get_metrics(self):
        """create one set of metric in each condition, 
        metric name can be shared across condition (to plot on the same plot in tensorboard)
        """
        metrics = {}
        for task in self.tasks:
            metrics[task]={}

            if task == 'triangle':
                for cond in self.conds:
                    pho_metrics = [metric_class(f"triangle_pho_{metric_name}") for metric_name, metric_class in self.METRICS_MAP[task]['pho'].items()]
                    sem_metrics = [metric_class(f"triangle_sem_{metric_name}") for metric_name, metric_class in self.METRICS_MAP[task]['sem'].items()]
                    metrics[task][cond] = [pho_metrics, sem_metrics]
            else:
                for cond in self.conds:
                    metrics[task][cond] = [metric_class(f"{task}_{metric_name}") for metric_name, metric_class in self.METRICS_MAP[task].items()]

        return metrics

    def _get_x(self, cond, input):
        return [self.testsets[cond][input]] * self.cfg.n_timesteps

    def _get_y(self, cond, output):
        return [self.testsets[cond][output]] * self.cfg.output_ticks

    def __call__(self, task, step):
        if task == "triangle":
            x = {cond: self._get_x(cond, 'ort') for cond in self.conds}
            y = {cond: [self._get_y(cond, 'pho'), self._get_y(cond, 'sem')] for cond in self.conds}

            model.set_active_task(task)
            [self.teststeps[task](x[cond], y[cond], model, self.metrics[task][cond]) for cond in self.conds]

            for cond in self.conds:
                with self.writers[cond].as_default():
                    [tf.summary.scalar(m.name, m.result(), step=step) for outputs in self.metrics[task][cond] for m in outputs]

        else:
            x_name, y_name = modeling.IN_OUT[task]
            x = {cond: self._get_x(cond, x_name) for cond in self.conds}
            y = {cond: self._get_y(cond, y_name) for cond in self.conds}

            model.set_active_task(task)
            [self.teststeps[task](x[cond], y[cond], model, self.metrics[task][cond]) for cond in self.conds]

            for cond in self.conds:
                with self.writers[cond].as_default():
                    [tf.summary.scalar(m.name, m.result(), step=step) for m in self.metrics[task][cond]]


strain = BaseTestSet(
    cfg, 
    name='strain', 
    conds=('hf_con_hi', 'hf_con_li', 'hf_inc_hi', 'hf_inc_li', 'lf_con_hi', 'lf_con_li', 'lf_inc_hi', 'lf_inc_li'),
    tasks=cfg.tasks
)


# Grain use two testset and combine results?


class GrainTestSet(BaseTestSet):
    METRICS_MAP = {
        'pho_pho':{'acc': metrics.PhoMultiAnsAccuracy},
        'sem_pho':{'acc': metrics.PhoMultiAnsAccuracy},
        'pho_sem':{'acc': metrics.RightSideAccuracy, 'sse': metrics.SumSquaredError},
        'sem_sem':{'acc': metrics.RightSideAccuracy, 'sse': metrics.SumSquaredError},
        'triangle':{
            'pho': {'acc': metrics.PhoMultiAnsAccuracy},
            'sem': {'acc': metrics.RightSideAccuracy, 'sse': metrics.SumSquaredError}
        }
    }

    def __init__(self, cfg, name, conds, tasks):
        super().__init__(cfg, name, conds, tasks)

    def _get_y(self, cond, output):
        return self.testsets[cond][output]

    @staticmethod
    def get_test_step(task):
        if task == 'triangle':
            @tf.function()
            def test_step(x, y, model, metrics):
                # Unpacking for easier access
                y_pred_pho, y_pred_sem = model(x)
                y_pho, y_sem = y
                metrics_pho, metrics_sem = metrics

                [m.update_state(tf.cast(y_pho, tf.float32), y_pred_pho[-1]) for m in metrics_pho]
                [m.update_state(tf.cast(y_sem, tf.float32), y_pred_sem[-1]) for m in metrics_sem]

        else:
            @tf.function()
            def test_step(x, y, model, metrics):
                y_pred = model(x)
                [m.update_state(y, y_pred[-1]) for m in metrics]
        return test_step



grain = GrainTestSet(
    cfg, 
    name='grain', 
    conds=('ambiguous', 'unambiguous'),
    tasks=('pho_pho', 'sem_pho')
)
