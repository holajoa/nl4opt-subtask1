from transformers import Trainer
from transformers.utils import logging


class MyTrainer(Trainer):
    def log(self, logs):
        """Overwrite original log method to log to external file."""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        # log to external file
        logging.info(output)
    