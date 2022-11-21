import sys
sys.path.insert(0, '..')

from BaseLogger import BaseLogger


class SyntheticExpLogger(BaseLogger):
    def __init__(self):
        super().__init__("syn-exp-logger")

    def log_param_tune(self, model_info, run_info):
        msg = (
            f"Optimization - "
            f"Graph idx: {run_info['graph_idx']}, "
            f"Weight decay: {run_info['weight_decay']}, "
            f"Dropout: {run_info['dropout']}, "
            f"Best Test Result: {run_info['result']:.4f}, "
            f"Training Loss: {run_info['loss']:.4f} "
        )
        self.logger.info(msg)

    def log_best_result(self, model_info, best_result_info):
        msg = (
            f"Optimization Result - "
            f"Model type: {model_info['model_type']}, "
            f"Graph Type: {model_info['graph_type']}, "
            f"Edge homo: {model_info['edge_homo']}, "
            f"Base dataset: {model_info['base_dataset']}, "
            f"Number of same edge: {model_info['num_edge_same']}, "
            f"Learning rate: {model_info['lr']}, "
            f"Weight decay: {best_result_info['weight_decay']}, "
            f"Dropout: {best_result_info['dropout']}, "
            f"Test Mean: {best_result_info['result']:.4f}, "
            f"Test Mean: {best_result_info['std']:.4f} "
        )
        self.logger.info(msg)

    def log_run(self, run_info):
        msg = (
            f"Graph {run_info['graph_idx']} Run Summary - "
            f"Test Mean: {run_info['result']:.4f} "
        )
        self.logger.info(msg)

    def log_record(self, model_info, record_info):
        msg = (
            f"Best Result - "
            f"Model type: {model_info['model_type']}, "
            f"Graph Type: {model_info['graph_type']}, "
            f"Edge homo: {model_info['edge_homo']}, "
            f"Base dataset: {model_info['base_dataset']}, "
            f"Number of same edge: {model_info['num_edge_same']}, "
            f"Learning rate: {model_info['lr']}, "
            f"Weight decay: {model_info['weight_decay']}, "
            f"Dropout: {model_info['dropout']}, "
            f"Test Mean: {record_info['result']:.4f}, "
            f"Test Mean: {record_info['std']:.4f} "
        )
        self.logger.info(msg)

