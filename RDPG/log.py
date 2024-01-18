from torch.utils.tensorboard import SummaryWriter
import os
import torch


class TensorBoardHandler:
    def __init__(self, log_dir="runs/experiment", log_dir_increment=True):
        """
        Initializes the TensorBoardHandler.

        Args:
            log_dir (str): Directory where to save the log files.
        """
        i = 1
        if log_dir_increment:
            while os.path.exists(log_dir):
                log_dir = f"runs/experiment_{i}"
                i += 1
    
        self.writer = SummaryWriter(log_dir)


    def log_scalar(self, tag, value, step):
        """
        Logs a scalar value.

        Args:
            tag (str): Data identifier
            value (float): Value to log
            step (int): Global step value to record
        """
        self.writer.add_scalar(tag, value, step)

    def log_model_graph(self, model, input_size):
        """
        Logs the model graph.

        Args:
            model (torch.nn.Module): Model to log
            input_size (tuple): Size of the input tensor
        """
        sample_input = torch.rand(input_size)  # Adjust input size to match your model
        self.writer.add_graph(model, sample_input)

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()
