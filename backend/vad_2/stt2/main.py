# main.py
from controller import STTController

if __name__ == "__main__":
    # Model choice: "small" per your selection. Use device="cpu" for now.
    controller = STTController(model_name="small", device="cpu")
    controller.run()