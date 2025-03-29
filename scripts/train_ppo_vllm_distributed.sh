#!/usr/bin/env bash

export VLLM_LOGGING_LEVEL=WARNING

DEVICES="0,1,2,3,4,5,6,7"

NUM_DEVICES=$(echo $DEVICES | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
    --nproc_per_node="$((NUM_DEVICES - 1))" \
    --master_port=12345 \
    recipes/full_ppo_vllm_distributed.py \
    --config-name="full_ppo_vllm_distributed" \
    config.pretrained_model_name_or_path="${CKPT_PATH}" \
    model._target_=transformers."${MODEL_CLASS_NAME}" \
    processor._target_=transformers.AutoProcessor.from_pretrained \
    processor.pretrained_model_name_or_path="${PROCESSOR_PATH}" \
    dataset._target_=maye.datasets.MathGenerationDataset \
    validation_dataset._target_=maye.datasets.MathGenerationDataset \
    test_dataset._target_=maye.datasets.MathGenerationDataset \
    collate_fn._target_=maye.utils.collate.collate_rlhf_vllm \
    validation_collate_fn._target_=maye.utils.collate.collate_generation_vllm \
    test_collate_fn._target_=maye.utils.collate.collate_generation_vllm \
    loss._target_=maye.rlhf.loss.PPOLoss \
    loss.epsilon_low="${EPSILON_LOW}" \
    loss.epsilon_high="${EPSILON_HIGH}" \
    loss.kl_loss_coeff="${KL_LOSS_COEFF}" \
    vllm.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
    dataset.dataset_path="${DATASET_PATH}" \
    validation_dataset.dataset_path="${VALIDATION_DATASET_PATH}" \
    test_dataset.dataset_path="${TEST_DATASET_PATH}" \
    dataset.use_chat_template="${USE_CHAT_TEMPLATE}" \
    generation_kwargs.temperature="${TEMPERATURE}" \
    optimizer.lr="${LR}" \
    compile=True \
    output_dir="${OUTPUT_DIR}" \
    train_vit="${TRAIN_VIT}" \
    train_connector="${TRAIN_CONNECTOR}" \
    train_llm="${TRAIN_LLM}" \
    batch_size="${BATCH_SIZE}" \
    forward_batch_size="${FORWARD_BATCH_SIZE}" \
    ppo_epochs="${PPO_EPOCH}" \
    ppo_batch_size="${PPO_BATCH_SIZE}" \
    num_epochs="${NUM_EPOCHS}" \
    gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    penalise_no_eos="${PENALISE_NO_EOS}" \
    reward_penalty="${REWARD_PENALTY}" \
    min_response_length="${MIN_RESPONSE_LENGTH}" \
    seed="${SEED}" \
    clip_grad_norm="${CLIP_GRAD_NORM}" \
    enable_activation_checkpointing="${ENABLE_ACTIVATION_CHECKPOINTING}" \
    enable_activation_offloading="${ENABLE_ACTIVATION_OFFLOADING}" \
    kl_reward_coeff="${KL_REWARD_COEFF}" \
    whiten_rewards="${WHITEN_REWARDS}" \
    gamma="${GAMMA}" \
    log_every_n_steps="${LOG_EVERY_N_STEPS}" \
    save_every_n_epochs="${SAVE_EVERY_N_EPOCHS}" \
    save_eval_files="${SAVE_EVAL_FILES}" \
    enable_lr_scheduler="${ENABLE_LR_SCHEDULER}" \
    metric_logger.project="${PROJECT}" \
    metric_logger.name="${TAG}" \
    metric_logger.tags=["${TAG}"] \
