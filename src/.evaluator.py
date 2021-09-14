import os
from abc import ABC, abstractmethod
import evaluate
import meta

class EvaluatorFactory:
    def __init__(
        self, 
        code_name:str, 
        batch_name:str=None, 
        tf_root:str=None,
        tau_override:float=None,
        ):

        if batch_name:
            cfg_json = os.path.join("models", batch_name, code_name, "model_config.json")
        else:
            cfg_json = os.path.join("models", code_name, "model_config.json")

        self.cfg = meta.Config.from_json(cfg_json)

        # Inject new setting to config
        self.cfg.tf_root = tf_root if tf_root is not None 
        self.cfg.output_ticks = 13

        if tau_override is not None:
            cfg.tau_original = cfg.tau
            cfg.tau = tau_override
            cfg.output_ticks = int(round(cfg.output_ticks * (cfg.tau_original / cfg.tau)))

    def get_evaluator(self, eval_type:str):

    @staticmethod
    def inject_settings(cfg: meta.Config, settings: dict) -> meta.Config:



class TarabanEvaluator:
    def __init__(self, code_name:str, batch_name:str=None, tf_root:str=None):
        self.code_name = code_name
        self.cfg = load_config(code_name)
        self.batch_name = batch_name
        self.tf_root = tf_root if tf_root is not None else './'
        self.board = board

    def evaluate(self, player):
        return evaluate.evaluate(self.board, player)