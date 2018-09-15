import PIL
from io import BytesIO
import tensorboardX as tb
from tensorboardX.summary import Summary


class TensorBoard(object):
    def __init__(self, model_dir):
        self.summary_writer = tb.FileWriter(model_dir)

    def scalar_summary(self, tag, value, step):
        summary= Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)
