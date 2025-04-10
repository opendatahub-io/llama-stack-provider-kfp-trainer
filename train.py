import os
import time
import uuid

from llama_stack_client import LlamaStackClient


client = LlamaStackClient(base_url=f"http://localhost:8321")

simpleqa_dataset_id = "huggingface::simpleqa1"
_ = client.datasets.register(
    purpose="post-training/messages",
    source={
        "type": "uri",
        "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
    },
    dataset_id=simpleqa_dataset_id,
)


from llama_stack_client.types import (
    post_training_supervised_fine_tune_params,
)

training_config = post_training_supervised_fine_tune_params.TrainingConfig(
    data_config=post_training_supervised_fine_tune_params.TrainingConfigDataConfig(
        batch_size=1,
        data_format="instruct",
        dataset_id=simpleqa_dataset_id,
        shuffle=True,
    ),
    gradient_accumulation_steps=1,
    max_steps_per_epoch=1,
    max_validation_steps=1,
    n_epochs=1,
    optimizer_config=post_training_supervised_fine_tune_params.TrainingConfigOptimizerConfig(
        lr=2e-5,
        num_warmup_steps=1,
        optimizer_type="adam",
        weight_decay=0.01,
    ),
)

from llama_stack_client.types.algorithm_config_param import LoraFinetuningConfig

algorithm_config = LoraFinetuningConfig(
    alpha=1,
    apply_lora_to_mlp=True,
    apply_lora_to_output=False,
    lora_attn_modules=['q_proj'],
    rank=1,
    type="LoRA",
)

job_uuid = f'test-job{uuid.uuid4()}'
training_model = os.environ["INFERENCE_MODEL"]

start_time = time.time()
response = client.post_training.supervised_fine_tune(
    job_uuid=job_uuid,
    logger_config={},
    model=training_model,
    hyperparam_search_config={},
    training_config=training_config,
    algorithm_config=algorithm_config,
    checkpoint_dir="null", # API claims it's not needed but - 400 if not passed.
)

print("Job: ", job_uuid)

while True:
    status = client.post_training.job.status(job_uuid=job_uuid)
    if not status:
        print("Job not found")
        break

    print(status)
    if status.status == "completed":
        break

    print("Waiting for job to complete...")
    time.sleep(5)

end_time = time.time()
print("Job completed in", end_time - start_time, "seconds!")

print('Artifacts:')
print(client.post_training.job.artifacts(job_uuid=job_uuid))

