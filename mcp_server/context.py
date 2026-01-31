class ModelContext:
    def __init__(self, model_name, param_count, input_shape):
        self.model_name = model_name
        self.param_count = param_count
        self.input_shape = input_shape

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "params": f"{self.param_count / 1e6:.2f}M",
            "input_shape": self.input_shape
        }

