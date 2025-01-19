import torch
from setfit import SetFitModel


class Classifier:
    """
    A classifier class for handling and managing machine learning models, specifically for SetFitModel.

    Attributes:
        model_path (str): The file path to the saved model.
        device (str): The computation device (either 'cuda' for GPU or 'cpu').
        label (str): The label associated with the classifier.
        model (SetFitModel or None): The loaded SetFitModel instance or None if not loaded.
    """

    def __init__(self, model_path, label):
        """
        Initializes the Classifier instance with the specified model path and label.

        Args:
            model_path (str): The path where the model is stored.
            label (str): The label associated with this classifier.
        """
        self.model_path = model_path
        # Determine if CUDA (GPU support) is available, else use CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label = label
        self.model = None  # Model is initially not loaded.

    def load_model(self, logger):
        """
        Loads the SetFitModel from the specified path and moves it to the appropriate device (GPU or CPU).

        Returns:
            SetFitModel: The loaded model.
        """
        # Load the model from the pre-trained path and move to the specified device (GPU or CPU).
        try:
            self.model = SetFitModel.from_pretrained(self.model_path).to(self.device)
            logger.info(f"{self.label} model successfully loaded!")
            return self.model
        
        except:
            return
    def get_model(self):
        """
        Retrieves the loaded model.

        Returns:
            SetFitModel or None: The currently loaded model or None if no model is loaded.
        """
        return self.model
