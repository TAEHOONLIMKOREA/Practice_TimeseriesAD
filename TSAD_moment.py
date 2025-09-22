from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-small", 
    model_kwargs={"task_name": "reconstruction"},
)
model.init()